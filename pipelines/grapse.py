import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from classes import Pipeline
from langchain.prompts import PromptTemplate
from argparse import ArgumentParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableMap
)
from langchain_core.messages import AIMessage, HumanMessage
import warnings
import tiktoken


warnings.filterwarnings("ignore")

# Load the environment variables used for evaluation
load_dotenv(dotenv_path='.env')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

__all__ = ["MainPipeline"]

def num_tokens_from_string(string: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Returns the number of tokens in a text string for a given model.
    """
    # For GPT-4 and GPT-3.5, "cl100k_base" is the typical encoding in tiktoken.
    # Adjust if your model differs
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(string)
    return len(tokens)

class MainPipeline(Pipeline):

    def __init__(self, model_name):
        self.model_name = model_name
        self.retrieved_docs = None

    def format_docs(self, docs):
        """
        Utility to format docs (a list of langchain.schema.Document)
        into a single string, also store them in self.retrieved_docs.
        """
        if self.retrieved_docs is None:
            self.retrieved_docs = [doc.page_content for doc in docs]
        else:
            self.retrieved_docs += [doc.page_content for doc in docs]
        return "\n\n".join(doc.page_content for doc in docs)

    def get_community_context(self, inputs):
        """
        Existing code to retrieve summary contexts from __Community__ nodes
        based on vector search. 
        """
        question = inputs["question"]
        query_vector = OpenAIEmbeddings().embed_query(question)
        cypher_query = f"""
        WITH $query_vector AS qv
        CALL (qv) {{
            WITH qv
            MATCH (n:__Community__)
            WHERE n.embedding IS NOT NULL
            WITH n, gds.similarity.cosine(n.embedding, qv) AS score
            ORDER BY score DESC
            LIMIT 5
            RETURN n AS node, score
        }}
        RETURN node{{.summary}} AS nodeProperties, score
        """

        records = self.graph.query(cypher_query, {"query_vector": query_vector})
        if self.retrieved_docs is None:
            self.retrieved_docs = [r["nodeProperties"]["summary"] for r in records]
        else:
            self.retrieved_docs += [r["nodeProperties"]["summary"] for r in records]

        return "\n\n".join([r["nodeProperties"]["summary"] for r in records])
    
    def get_global_context(self, local_docs, question):
        """
        A new method for 'global' retrieval that expands beyond the locally
        retrieved docs. Here, we do a simple example of:
          - Extract doc IDs from the local retrieval
          - Match connected docs via edges in the graph
          - Optionally compute embeddings to filter or re-rank
        Adjust or replace this logic to suit how your graph is structured.
        """

        # If your locally retrieved Documents include doc.metadata["id"],
        # we can gather them here:
        local_doc_ids = []
        for doc in local_docs:
            if "id" in doc.metadata:
                local_doc_ids.append(doc.metadata["id"])

        if not local_doc_ids:
            # If we have no doc IDs, we can’t do a direct graph expansion:
            return ""

        query_vector = OpenAIEmbeddings().embed_query(question)

        # Example Cypher that:
        #  (1) UNWIND the local doc IDs
        #  (2) MATCH related docs via edges (e.g. :LINKS_TO or :MENTIONS)
        #  (3) Optional: measure embedding similarity with question
        #  (4) Return top-scoring expansions
        # *** You must adapt the MATCH pattern & relationship to your graph ***
        cypher_query = """
            WITH $local_doc_ids AS localIds, $query_vector AS qv
            UNWIND localIds AS docId
            MATCH (d:Document {id: docId})-[*1..2]-(g:Document)
            WHERE g.embedding IS NOT NULL AND g.id <> docId
            WITH g, gds.similarity.cosine(g.embedding, qv) AS score
            ORDER BY score DESC
            LIMIT 5
            RETURN g.text AS text, score
        """

        records = self.graph.query(cypher_query, {
            "local_doc_ids": local_doc_ids,
            "query_vector": query_vector
        })

        # Create Document objects on the fly (or you can do something else).
        from langchain.schema import Document
        global_docs = []
        for r in records:
            global_docs.append(
                Document(
                    page_content=r["text"],
                    metadata={"score": r["score"]}   # store the similarity if needed
                )
            )

        # Format these retrieved global docs so we can combine them in the final context:
        return self.format_docs(global_docs)

    def setup(self):
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(temperature=0.0, model_name=self.model_name)

        # Define your custom retrieval query
        retrieval_query = """
            MATCH (n:Document)
            WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
            RETURN n.text AS text, 
                {id: n.id, source_paper_name: n.source_paper_name, source_authors: n.source_authors} AS metadata, 
                score
            ORDER BY score DESC
            LIMIT $k
            """

        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="vector",
            node_label="Document",
            text_node_properties=["text", "description"],
            embedding_node_property="embedding",
            retrieval_query=retrieval_query
        )

        # create vector index for communities
        self.community_vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="vector",
            node_label="__Community__",
            text_node_properties=["summary"],
            embedding_node_property="embedding"
        )

        # Create retrievers for both the document and community vector indexes
        self.vector_retreiver = self.vector_index.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={'score_threshold': 0.5}
        )

        def combined_retriever_fn(inputs):
            question = inputs["question"]
            # 1) Local (Document) retrieval:
            local_docs = self.vector_retreiver.invoke(question)

            # 2) Global (Graph expansion) retrieval:
            global_context = self.get_global_context(local_docs, question)

            # 3) Community retrieval:
            community_context = self.get_community_context(inputs)

            # Combine everything into one big context string
            combined_context = (
                self.format_docs(local_docs) 
                + "\n\n" 
                + global_context
                + "\n\n"
                + community_context
            )

            # --- Check token size! ---
            max_tokens = 128_000  # your gpt-4o-mini limit
            # keep some buffer for the system prompt, question, and LLM answer
            safe_limit = max_tokens - 2000

            current_tokens = num_tokens_from_string(combined_context, "gpt-4o-mini")
            #print(f"Current tokens: {current_tokens}")
            if current_tokens > safe_limit:
                print(f"WARNING: Truncating context to {safe_limit} tokens from {current_tokens} tokens")
                # naive truncation approach
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(combined_context)
                truncated_tokens = tokens[:safe_limit]
                truncated_text = encoding.decode(truncated_tokens)
                combined_context = truncated_text


            return {
                "context": combined_context,
                "question": question
            }

        combined_retriever = RunnableLambda(combined_retriever_fn)

        # Create a single prompt for the combined context
        combined_prompt = ChatPromptTemplate.from_messages([
            ("human", '''You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. Keep the answer concise and to the point.\n
            Question: {question}\n
            Context: {context}\n
            Answer:'''),
        ])

        # Create a single chain for the combined context
        self.combined_rag_chain = (
            combined_retriever
            | combined_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_answer(self, question):
        input_data = {"question": question}
        response = self.combined_rag_chain.invoke(input_data)
        return response, self.retrieved_docs


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("-m", "--model", type=str, required=False, default="gpt-4o-mini", help="Model to use for the answer")
    args = parser.parse_args()
    pipeline = MainPipeline(args.model)
    pipeline.retrieved_docs = None
    pipeline.setup()
    answer, context = pipeline.get_answer(args.question)
    print("Answer:\n", answer)
    #print("\nRetrieved Docs:\n", context)