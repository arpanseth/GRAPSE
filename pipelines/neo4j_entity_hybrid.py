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
warnings.filterwarnings("ignore")
import re

# Load the environment variables used for evaluation
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

__all__ = ["MainPipeline"]

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the Author, Algorithm, Software, Solver, Formula, Institution, and Benchmark entities that" 
        "appear in the text",
    )

class MainPipeline(Pipeline):

    def format_docs(self, docs):
        self.retrieved_docs = [doc.page_content for doc in docs]
        return "\n\n".join(doc.page_content for doc in docs)

    def setup(self):
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(temperature=0, model_name=self.model_name)

        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="vector",
            node_label="Document",
            text_node_properties=["text", "description"],
            embedding_node_property="embedding"
        )
        
        # Condense a chat history and follow-up question into a standalone question
        self._template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""  # noqa: E501
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self._template)


        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting Author, Algorithm, Software, Solver, Formula, Institution, and Benchmark entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        self.entity_chain = self.prompt | self.llm.with_structured_output(Entities)
        self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        # Define search query 
        self._search_query = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),  # Condense follow-up question and chat into a standalone_question
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | ChatOpenAI(temperature=0)
                | StrOutputParser(),
            ),
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x : x["question"]),
        )

        # Define Context pipeline
        self.context_pipeline = self._search_query | self.retriever
        self.context_pipeline_with_list = self._search_query | self.retriever_with_list

        self.template = """Answer the question based on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # Build a RunnableMap that has two branches:
        #    - "answer": a pipeline that merges {context, question} then calls prompt->LLM->parse
        #    - "context": returns the same retrieval used in "answer"
        self.chain = RunnableMap(
            {
                "answer": (
                    # Combine question + context in parallel
                    RunnableParallel(
                        {
                            "context": self.context_pipeline, 
                            "question": RunnablePassthrough()
                        }
                    )
                    # Then pass them to prompt -> LLM -> parser
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
                ),
                "context": self.context_pipeline_with_list
            }
        )

    def remove_lucene_chars_preserving_hyphen(self, input_str: str) -> str:
        """
        Removes Lucene special characters except hyphens.
        Escapes hyphens to prevent them from being treated as operators.
        """
        lucene_special_chars = r'([+\!(){}\[\]^"~*?:\\/])'
        cleaned = re.sub(lucene_special_chars, '', input_str)
        # Escape hyphens
        cleaned = re.sub(r'(-)', r'\\\1', cleaned)
        return cleaned

    def generate_full_text_query(self, input_str: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by preserving hyphenated words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.
        """
        cleaned_input = self.remove_lucene_chars_preserving_hyphen(input_str)
        words = [el for el in cleaned_input.split() if el]
        if not words:
            # If the input was entirely stripped or empty
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        return full_text_query.strip()
    
    # Fulltext index query
    def structured_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
                YIELD node, score
                CALL (node) {
                    MATCH (node)-[r:!MENTIONS]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    MATCH (node)<-[r:!MENTIONS]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output
                LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            for el in response:
                if isinstance(el['output'], str):
                    result += "\n" + el['output']
                elif isinstance(el['output'], list):
                    result += "\n" + "".join(el['output'])    
        return result
    
    def retriever(self, question: str):
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        final_data = f"""Structured data:

            {structured_data}

            Unstructured data:

            {"#Document ". join(unstructured_data)}
        """
        return final_data

    def retriever_with_list(self, question: str):
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        unstructured_data.append(structured_data)
        return unstructured_data

    
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def get_answer(self, question):
        response = self.chain.invoke({"question": question})
        return response['answer'], response['context']


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("-m", "--model", type=str, required=False, default="gpt-4o-mini", help="Model to use for the answer")
    args = parser.parse_args()
    pipeline = MainPipeline(args.model)
    pipeline.setup()
    answer, context = pipeline.get_answer(args.question)
    print(answer)
    #print(context)