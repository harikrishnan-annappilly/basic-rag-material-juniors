import sys
import asyncio

# --- WINDOWS FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGEngine, PGVectorStore


def test_modern_hyder():
    llm = ChatOllama(model="llama3.1", temperature=0.3)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    CONNECTION_STRING = "postgresql+psycopg://postgres:mysecretpassword@localhost:5431/postgres"
    engine = PGEngine.from_connection_string(CONNECTION_STRING)
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name="document_embeddings",
        embedding_service=embedding_model,
    )
    base_retriver = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt_template = """Please write a short, informative paragraph answering the following question.
It does not need to be perfectly accurate, but it should sound like a formal technical document.
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    custom_llm = prompt | llm | StrOutputParser()
    hyde_retriver = prompt | llm | StrOutputParser() | base_retriver

    test_question = "What is the token expiration lifetime for the access token?"
    print(f"\nExecuting pipeline for: '{test_question}'\n")

    response = custom_llm.invoke({"question": test_question})
    print("response from LLM")
    print(response)
    print("-" * 20)
    docs = hyde_retriver.invoke({"question": test_question})

    print("--- Retrieved Documents using Modern HyDE ---")
    for i, doc in enumerate(docs):
        print(f"Rank {i+1}: {doc}")


if __name__ == "__main__":
    test_modern_hyder()
