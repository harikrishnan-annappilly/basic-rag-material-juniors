import sys
import asyncio

# --- WINDOWS FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.documents import Document


def test_metadata_filtering():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    CONNNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    engine = PGEngine.from_connection_string(CONNNECTION_STRING)
    TABLE_NAME = "document_embeddings"

    store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embedding_model,
    )

    docs = [
        Document(
            page_content="Admins can reset passwords using the master dashboard.",
            metadata={"role": "admin", "topic": "security"},
        ),
        Document(
            page_content="Users can reset passwords by clicking 'Forgot Password' on the login screen.",
            metadata={"role": "user", "topic": "security"},
        ),
    ]
    # store.add_documents(docs)
    print("Searching with a metadata filter...")
    query = "How do I change password?"

    results = store.similarity_search(
        query=query,
        k=3,
        filter={"role": "user"},
    )
    for i, result in enumerate(results):
        print(f"---- Result {i+1} ----")
        print(result)
        print("-" * 20)


if __name__ == "__main__":
    test_metadata_filtering()
