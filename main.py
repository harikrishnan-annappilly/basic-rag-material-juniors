import os
import logging

from src.retrieval.retriever import load_existing_db, retrive_info
from src.prompts.prompt_templates import get_template
from src.llm.llm_client import get_llm

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_rag_pipeline(user_question):
    """
    Orchestrates the whole RAG workflow:
    Loads DB -> Retrieves Context -> Formats Prompt -> Generates Answer
    """
    logging.info(f"User asked question: {user_question}")
    print("Processing your request")
    try:
        db = load_existing_db()
        relevant_docs = retrive_info(user_question, db)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        logging.info("Successfully retrieved relevant context from vector database")

        prompt_template = get_template()
        formatted_prompt = prompt_template.format(context=context_text, question=user_question)
        llm = get_llm()
        print("Thinking...")
        response = llm.invoke(formatted_prompt)
        logging.info("Successfully generated response form llm")
        return response.content
    except Exception as ex:
        error_msg = f"Something went wrong: {ex}"
        logging.error(error_msg)
        return f"Sorry, something went wrong. Check logs"


if __name__ == "__main__":
    print("=== Welcome to your local RAG assistant ===")
    query = "Who is Hari?"
    answer = run_rag_pipeline(query)
    print("===Assistant answer start ===")
    print(answer)
    print("===Assistant answer end ===")
