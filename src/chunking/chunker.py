from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(extracted_pages):
    """
    This function takes the full page of text and slices them
    into smaller, overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_pages)
    return chunks


if __name__ == "__main__":
    import sys
    import os

    # Adding the parent directory to the path so we can import from ingestion
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.ingestion.loader import load_document

    test_file = "sample.pdf"
    try:
        print("Loading document")
        pages = load_document(test_file)

        print("Chunking document")
        chunks = split_text_into_chunks(pages)

        print(f"Success! sliced documents into {len(chunks)} chunks")

        print("\n--- Preview of chunk 1 Start ---\n")
        print(chunks[0].page_content)
        print("\n--- Preview of chunk 1 End ---\n")
    except Exception as ex:
        print(f"Something went wrong: {ex}")
