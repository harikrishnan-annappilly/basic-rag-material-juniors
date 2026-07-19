from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from main import run_rag_pipeline

# 1. Initialize our API "waiter"
app = FastAPI(title="My Local RAG Assistant API", description="An API to ask questions to my private documents.")


# 2. Define the shape of the data we expect from the user
# This ensures the user sends a "question" that is a string of text
class QueryRequest(BaseModel):
    question: str


# 3. Create an "Endpoint" (a specific URL path where the waiter listens)
@app.post("/ask")
def ask_assistant(request: QueryRequest):
    """
    This endpoint takes a question, sends it to our RAG pipeline,
    and returns the answer.
    """
    print(f"\n🌐 API Received a question: '{request.question}'")

    if not request.question or request.question.strip() == "":
        raise HTTPException(status_code=400, detail="You can not ask empty question.")

    try:
        # We hand the question to our master chef (the function from Day 10)
        final_answer = run_rag_pipeline(request.question)

        # We return the answer wrapped in a clean JSON format
        return {"question_asked": request.question, "assistant_answer": final_answer}
    except Exception as ex:
        print(f"CRITICAL ERROR {str(ex)}")
        raise HTTPException(status_code=500, detail="AI assitant currently unavailable or encoutered an error")
