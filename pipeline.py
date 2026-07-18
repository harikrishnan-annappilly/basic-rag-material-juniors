from src.ingestion.loader import load_document
from src.chunking.chunker import split_text_into_chunks
from src.vectordb.vector_store import create_vector_store
from src.utils.helpers import load_config


def run_week1_pipeline(file_path):
    config = load_config()

    pages = load_document(file_path)

    chunks = split_text_into_chunks(pages)

    db = create_vector_store(chunks)

    print(f"✅ Success! Data processed and stored in {config['db_name']} directory.")
    print(f"More details: {config}")


if __name__ == "__main__":
    run_week1_pipeline("sample.pdf")
