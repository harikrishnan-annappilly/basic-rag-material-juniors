from langchain_community.document_loaders import PyPDFLoader


def load_document(file_path):
    """
    This function will take a file path of a pdf file.
    Reads the document and returns the pages as raw text.
    """
    loader = PyPDFLoader(file_path=file_path)

    extracted_pages = loader.load()

    return extracted_pages


if __name__ == "__main__":
    test_file_path = "sample.pdf"

    try:
        pages = load_document(test_file_path)
        print(f"Successfully loading {len(pages)} pages!")

        print("----- Page 1 -----")
        print(pages[0].page_content[:200])
    except Exception as ex:
        print(f"Oops, something went wrong: {ex}")
