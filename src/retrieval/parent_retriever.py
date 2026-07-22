import sys
import asyncio

# --- WINDOWS FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -------------------

from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever


def test_parent_retriver():
    base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    CONNNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    TABLE_NAME = "child_embeddings"
    engine = PGEngine.from_connection_string(CONNNECTION_STRING)
    try:
        engine.init_vectorstore_table(table_name=TABLE_NAME, vector_size=384)
        print("Table created")
    except:
        print("Table already exists")

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=base_embeddings,
    )

    doc_store = InMemoryStore()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    retriver = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    docs = [Document(page_content="""PostgreSQL is a powerful, open-source object-relational database system. 
It has more than 35 years of active development. 
Because of its architecture, it has earned a strong reputation for reliability, data integrity, and correctness.
For AI applications, the pgvector extension allows PostgreSQL to store and query mathematical embeddings.
When developers use pgvector, they can perform similarity searches directly inside SQL using the <=> operator.""")]

    retriver.add_documents(docs)

    test_query = "What operator is used for similarity search"

    retrived_docs = retriver.invoke(test_query)
    print("\n--- Retrieved Parent Document ---")
    print(retrived_docs[0].page_content)


if __name__ == "__main__":
    test_parent_retriver()
