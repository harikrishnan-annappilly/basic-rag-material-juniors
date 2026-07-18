from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedder():
    """
    This function sets up and returns embedding model
    """
    model_name = "all-MiniLM-L6-v2"

    print(f"Loading embedding model {model_name}")
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    return embedder


if __name__ == "__main__":
    import sys
    import os

    # Adding the parent directory to the path so we can import from ingestion
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.ingestion.loader import load_document
    from src.chunking.chunker import split_text_into_chunks

    test_file = "sample.pdf"

    try:
        pages = load_document(test_file)
        print("Ingestion success")
        chunks = split_text_into_chunks(pages)
        print("Chunking success")

        my_embedder = get_embedder()
        print("Converting text to numbers")
        vector = my_embedder.embed_query(chunks[1].page_content)

        print(f"\nSuccess! The text was converted into a list of {len(vector)} numbers.")
        print("\n--- Preview of the first 5 numbers in the vector ---")
        print(vector[:5])
    except Exception as ex:
        print(f"Something went wrong: {ex}")
