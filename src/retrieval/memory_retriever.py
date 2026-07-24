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
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever


def test_conversational_memory():
    print("Initializing Database & LLM...")
    llm = ChatOllama(model="llama3.1", temperature=0.0)
    base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    CONNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    vector_store = PGVectorStore.create_sync(
        engine=engine, table_name="document_embeddings", embedding_service=base_embeddings
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("Building the History-Aware Prompt...")
    # 1. We instruct the LLM to act strictly as a question re-writer, NOT an answerer.
    contextualize_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=contextualize_prompt,
    )

    fake_chat_history = [
        HumanMessage(content="What is the expiration time for the access token?"),
        AIMessage(content="The access token expires in 15 minutes."),
    ]
    follow_up_question = "Can I extend it?"

    retrieved_docs = history_aware_retriever.invoke(
        {
            "chat_history": fake_chat_history,
            "input": follow_up_question,
        },
    )
    print("--- Retrieved Documents based on Chat History ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"Rank {i+1}: {doc}...")


if __name__ == "__main__":
    test_conversational_memory()
