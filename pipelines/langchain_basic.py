import os, sys
from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from argparse import ArgumentParser
from classes import Pipeline
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

__all__ = ["MainPipeline"]
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class MainPipeline(Pipeline):
    
    def format_docs(self, docs):
        self.retrieved_docs = [doc.page_content for doc in docs]
        return "\n\n".join(doc.page_content for doc in docs)

    def setup(self):
        self.llm = ChatOpenAI(model=self.model_name)
        # get path to chroma db
        chroma_db_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'chroma_db')
        self.vectorstore = Chroma(persist_directory=chroma_db_path, 
                             embedding_function=OpenAIEmbeddings())
        self.vectorstore.get() 
        self.retreiver = self.vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.3})
        self.prompt = ChatPromptTemplate.from_messages([
                        ("human", '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
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
    print(pipeline.get_answer(args.question))