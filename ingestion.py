import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()


def ingest_txt_file(
    file_path: str | Path = "mediumblog.txt",
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> PineconeVectorStore:
    """Read a text file, split it into chunks, embed it, and store it in Pinecone."""
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    index_name = os.getenv("INDEX_NAME")
    if not index_name:
        raise RuntimeError("INDEX_NAME must be set in the environment")

    documents = UnstructuredLoader(file_path=str(path)).load()
    if not documents or not any(document.page_content.strip() for document in documents):
        raise ValueError(f"Text file is empty: {path}")

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.split_documents(documents)

    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        index_name=index_name,
    )
    return vector_store


if __name__ == "__main__":
    print("Ingesting...")
    ingest_txt_file()
    print("Ingestion complete.")
