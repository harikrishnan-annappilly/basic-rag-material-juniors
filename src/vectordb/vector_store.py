import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_chroma import Chroma
from src.embeddings.embedder import get_embedder
from src.utils.helpers import load_config


def create_vector_store(chunks):
    """
    This function takes the chunks, creates vectors for all of them.
    Then stores them in local chroma db
    """
    config = load_config()
    my_embedder = get_embedder()
    presistent_directory = config["db_name"]
    print(f"Deleting existing DB")
    shutil.rmtree(presistent_directory)
    print(f"Storing {len(chunks)} chunks in to Vector DB")

    vectore_store = Chroma.from_documents(
        documents=chunks,
        embedding=my_embedder,
        persist_directory=presistent_directory,
    )
    return vectore_store


if __name__ == "__main__":
    from src.ingestion.loader import load_document
    from src.chunking.chunker import split_text_into_chunks

    test_file = "sample.pdf"

    try:
        pages = load_document(test_file)
        chunks = split_text_into_chunks(pages)
        db = create_vector_store(chunks)

        print("Success! Database has been created and saved")
    except Exception as ex:
        print(f"Something went wrong: {ex}")
