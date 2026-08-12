import os
from dotenv import load_dotenv

from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    print(os.environ.get("LANGSMITH_API_KEY"))
    print(os.environ.get("PINECONE_API_KEY"))
