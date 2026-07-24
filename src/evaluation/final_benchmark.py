import sys
import asyncio
import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.chain.conversational_rag import build_modern_lcel_rag


def run_end_to_end_eval():
    print("Initializing the AI Judge...")
    judge_model = OllamaModel(model="llama3.1", temperature=0.0)

    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    print("Booting up the LCEL Pipeline...")
    rag_chain = build_modern_lcel_rag()
    chat_history = []

    q1 = "What is the token expiration lifetime for the access token?"
    print(f"\n[Turn 1] User: {q1}")
    r1 = rag_chain.invoke({"input": q1, "chat_history": chat_history})

    chat_history.append(("human", q1))
    chat_history.append(("ai", r1))

    q2 = "Can I extend it?"
    print(f"[Turn 2] User: {q2}")
    r2 = rag_chain.invoke({"input": q2, "chat_history": chat_history})
    print(f"[Turn 2] AI: {r2}\n")

    print("Sending the final interaction to the AI Judge...")

    test_case = LLMTestCase(input="Can I extend the access token?", actual_output=r2)

    results = evaluate(test_cases=[test_case], metrics=[relevancy_metric])
    print(results)
    print("✅ Final Benchmark Complete!")


if __name__ == "__main__":
    run_end_to_end_eval()
