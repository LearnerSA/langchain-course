import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Answer the question using only the context below.
If the answer is not present in the context, say that you do not know.

Context:
{context}

Question:
{question}
""",
        )
    ]
)


def format_context(documents) -> str:
    """Format retrieved documents into the context used by the prompt."""
    return "\n\n".join(
        f"Document {index + 1}:\n{document.page_content}"
        for index, document in enumerate(documents)
    )


def answer_with_rag(question: str, *, k: int = 4) -> str:
    """Retrieve relevant documents and answer the question using their context."""
    index_name = os.getenv("INDEX_NAME")
    if not index_name:
        raise RuntimeError("INDEX_NAME must be set in the environment")

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings(),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_documents = retriever.invoke(question)
    context = format_context(retrieved_documents)

    messages = RAG_PROMPT.format_messages(context=context, question=question)

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    response = llm.invoke(messages)
    return response.content


def main() -> None:
    question = "What is Pinecone in Machine Learning?"
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))

    print("Option 0 : Raw LLM Invocation (No RAG)")
    raw_response = llm.invoke([HumanMessage(content=question)])
    print(raw_response.content)

    print("\n" + "=" * 80 + "\n")
    print("Option 1 : RAG (Manual Invocation, No LCEL)")
    rag_response = answer_with_rag(question)
    print(rag_response)


if __name__ == "__main__":
    main()
