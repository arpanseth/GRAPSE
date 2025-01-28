import os, sys
from dotenv import load_dotenv

# Load the environment variables used for evaluation
load_dotenv(dotenv_path='.env')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the documents
print("Loading documents...")
path = "./papers/"
loader = DirectoryLoader(path, glob="**/*.pdf")
docs = loader.load()

# Split the documents into chunks AND generate the Chroma index
print("Generating Chroma index...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory='./output/chroma_db')


