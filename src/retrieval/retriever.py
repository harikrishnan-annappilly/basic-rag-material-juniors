import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_chroma import Chroma
from src.embeddings.embedder import get_embedder


def load_existing_db():
    """
    Loads our previously saved database from the folder
    """
    my_embedder = get_embedder()
    persistent_directory = "./chroma_db"

    print("Loading exiting vector database")

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=my_embedder,
    )
    return db


def retrive_info(query: str, db: Chroma):
    """
    Takes one user query, searches in db, returns the top 3 matches.
    """
    print(f"Searching for query :{query}")
    results = db.similarity_search(query, k=3)
    return results


if __name__ == "__main__":
    try:
        my_db = load_existing_db()

        test_query = "Who is Hari?"
        top_chunks = retrive_info(test_query, my_db)

        print(f"Found {len(top_chunks)} relevant chunks!")

        for i, chunk in enumerate(top_chunks):
            print(f"--- Match {i+1} ---")
            print(chunk.page_content)
            print("-" * 20 + "\n")
    except Exception as ex:
        print(f"Something went wrong: {ex}")
