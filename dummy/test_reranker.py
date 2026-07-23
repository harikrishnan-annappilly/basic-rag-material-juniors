# src/retrieval/test_reranker.py
import sys
import asyncio

# --- WINDOWS FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document


class ScoredCrossEncoder(CrossEncoderReranker):
    def compress_documents(self, documents, query, callbacks=None):
        scores = self.model.score([(query, doc.page_content) for doc in documents])

        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = float(score)

        documents.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        return documents[: self.top_n]


def test_cross_encoder():
    print("Initializing Database (Stage 1)...")
    base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    CONNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    vector_store = PGVectorStore.create_sync(
        engine=engine, embedding_service=base_embeddings, table_name="document_embeddings"
    )
    base_retriver = vector_store.as_retriever(search_kwargs={"k": 10})
    print("initializeign cross encoder")
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    reranker = ScoredCrossEncoder(model=cross_encoder, top_n=3)  # For getting the relevancy score
    compression_retriver = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriver,
    )

    test_query = "What is the exact expiration time for the access token?"
    print(f"\nExecuting Pipeline for: '{test_query}'\n")

    final_docs = compression_retriver.invoke(test_query)
    print("--- Final Re-Ranked Documents ---")
    for i, doc in enumerate(final_docs):
        # We can actually look at the exact score the Cross-Encoder assigned it!
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"Rank {i+1} | Score: {score} | {doc.page_content[:60]}...")


if __name__ == "__main__":
    test_cross_encoder()
