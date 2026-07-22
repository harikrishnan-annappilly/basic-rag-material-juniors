import sys
import asyncio

# --- WINDOWS FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)


def setup_multi_query_retriver():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    CONNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    engine = PGEngine.from_connection_string(CONNECTION_STRING)
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embedding_model,
        table_name="document_embeddings",
    )

    llm = ChatOllama(model="llama3.1", temperature=0)

    advanced_retriver = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )
    original_question = "What is the token expiration lifetime for the access token?"

    fused_docs = advanced_retriver.invoke(original_question)

    print("\n--- Final Fused Results ---")
    for i, doc in enumerate(fused_docs):
        print("-" * 20)
        print(f"Rank {i+1}: {doc.page_content}")
        print("-" * 20)


if __name__ == "__main__":
    setup_multi_query_retriver()
