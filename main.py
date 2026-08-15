import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.environ["INDEX_NAME"]
EMBEDDINGS = OpenAIEmbeddings()
VECTOR_STORE = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=EMBEDDINGS,
)
RETRIEVER = VECTOR_STORE.as_retriever(search_kwargs={"k": 4})
LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))

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


RAG_CHAIN = (
    {
        "context": RETRIEVER | format_context,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | LLM
    | StrOutputParser()
)


def answer_with_rag(question: str) -> str:
    """Retrieve relevant documents and answer the question using their context."""
    retrieved_documents = RETRIEVER.invoke(question)
    context = format_context(retrieved_documents)

    messages = RAG_PROMPT.format_messages(context=context, question=question)
    response = LLM.invoke(messages)
    return response.content


def answer_with_rag_lcel(question: str) -> str:
    """Answer a question with the same RAG flow composed using LCEL."""
    return RAG_CHAIN.invoke(question)


def main() -> None:
    question = "What is Pinecone in Machine Learning?"

    print("\n" + "=" * 90 + "\n")

    print("Option 0 : Raw LLM Invocation (No RAG)")
    raw_response = LLM.invoke([HumanMessage(content=question)])
    print(raw_response.content)

    print("\n" + "=" * 90 + "\n")
    print("Option 1 : RAG (Manual Invocation, No LCEL)")
    rag_response = answer_with_rag(question)
    print(rag_response)

    print("\n" + "=" * 90 + "\n")
    print("Option 2 : RAG with LCEL")
    rag_lcel_response = answer_with_rag_lcel(question)
    print(rag_lcel_response)


if __name__ == "__main__":
    main()
