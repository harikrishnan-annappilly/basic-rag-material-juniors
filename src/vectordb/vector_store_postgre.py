import sys
import asyncio
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# src/vectordb/vector_store.py
from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


from src.ingestion.loader import load_document
from src.chunking.chunker import split_text_into_chunks


def run_enterprise_ingestion():
    print("Loading embedding model...")
    # 1. We use our standard embeddings.
    # The MiniLM-L6-v2 model outputs vectors that are exactly 384 dimensions long.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    VECTOR_SIZE = 384

    print("Connecting to PostgreSQL...")
    # 2. Notice the "+psycopg" in the connection string!
    # This tells Python to use the modern, high-performance database driver.
    CONNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"

    # 3. Create the database engine
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    TABLE_NAME = "document_embeddings"

    print("Configuring the vector table...")
    # 4. LangChain will execute the SQL to create your table and configure
    # the vector dimensions automatically if the table doesn't already exist.
    try:
        engine.init_vectorstore_table(table_name=TABLE_NAME, vector_size=VECTOR_SIZE)
    except:
        print("Tables already exists")

    # 5. Initialize the actual Vector Store object
    store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embedding_model,
    )

    print("Inserting test documents...")
    # 6. We simulate inserting semantically chunked documents
    docs = [
        Document(
            page_content="The access token has an expiration lifetime of exactly 15 minutes.",
            metadata={"topic": "security"},
        ),
        Document(
            page_content="The database uses PostgreSQL with the pgvector extension.",
            metadata={"topic": "database"},
        ),
    ]

    # 7. LangChain handles converting text -> vector -> SQL insert automatically!
    pages = load_document("sample.pdf")
    chunks = split_text_into_chunks(pages)
    print(f"Adding {len(chunks)} to the DB")
    store.add_documents(chunks)

    print("✅ Documents successfully ingested into PostgreSQL!")


if __name__ == "__main__":
    try:
        run_enterprise_ingestion()
    except Exception as e:
        print(f"Database error: {e}")
