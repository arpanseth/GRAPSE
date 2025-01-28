import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
#from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from classes import Pipeline
from argparse import ArgumentParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load the environment variables used for evaluation
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

from langchain_openai import ChatOpenAI

__all__ = ["MainPipeline"]

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

        self.custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="You are an assistant helping with retrieval. Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        self.retreiver = self.vector_index.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.3})
        self.prompt = ChatPromptTemplate.from_messages([
                        ("human", '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
                        Question: {question} 
                        Context: {context} 
                        Answer:'''),
                        ])
        
        self.rag_chain = (
             {"context": self.retreiver | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        
        # self.qa_chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",  # Combines retrieved documents into the input for LLM
        #     retriever=self.vector_index.as_retriever(k=10),
        #     chain_type_kwargs={"prompt": self.custom_prompt}  # Custom prompt
        # )
    
    def get_answer(self, question):
        response = self.rag_chain.invoke(question)
        return response, self.retrieved_docs


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