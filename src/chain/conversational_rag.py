import sys
import asyncio

# --- WINDOWS FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_modern_lcel_rag():
    print("1. Initializing Models & Vector Store...")
    llm = ChatOllama(model="llama3.1", temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    CONNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    vector_store = PGVectorStore.create_sync(
        engine=engine, table_name="document_embeddings", embedding_service=embeddings
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    contextualize_system_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed or return it as is."
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                contextualize_system_prompt,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_rewriter = contextualize_prompt | llm | StrOutputParser()

    qa_system_prompt = "You are an expert enterprise assistant. Use the following pieces of retrieved context to answer the user's question. If you do not know the answer, say that you don't know. Keep your answer clear, accurate, and concise.\n\nContext:\n{context}"
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def retrive_documents(inputs):
        chat_history = inputs.get("chat_history", [])

        if chat_history:
            search_query = question_rewriter.invoke(inputs)
        else:
            search_query = inputs["input"]
        docs = base_retriever.invoke(search_query)
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = RunnablePassthrough.assign(context=retrive_documents) | qa_prompt | llm | StrOutputParser()
    return rag_chain


def run_chat_session():
    rag_chain = build_modern_lcel_rag()
    chat_history = []
    print("\n--- Starting Pure LCEL Conversation Session ---")

    # Turn 1
    query_1 = "What is the token expiration lifetime for the access token?"
    print(f"\nUser: {query_1}")

    response_1 = rag_chain.invoke({"input": query_1, "chat_history": chat_history})
    print(f"AI: {response_1}")

    chat_history.extend([HumanMessage(content=query_1), AIMessage(content=response_1)])

    query_2 = "Can I extent it?"
    print(f"\nUser: {query_2}")
    response_2 = rag_chain.invoke({"input": query_2, "chat_history": chat_history})
    print(f"AI: {response_2}")


if __name__ == "__main__":
    run_chat_session()
