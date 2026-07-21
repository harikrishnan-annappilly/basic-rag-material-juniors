from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def test_semantic_splitting():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80,
    )

    sample_text = (
        "The access token expires in exactly 15 minutes. This is critical for security. "
        "The database uses PostgreSQL. Vector indexes are built using HNSW."
    )
    print("---- Full Text ----")
    print(sample_text)
    print("-------------------")

    chunks = text_splitter.split_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"---- Chunk {i+1} ----")
        print(chunk)
        print(f"---------------------")


if __name__ == "__main__":
    test_semantic_splitting()
