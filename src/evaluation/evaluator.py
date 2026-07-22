from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import OllamaModel

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from main import run_rag_pipeline


def run_baseline_evaluation():
    judge_model = OllamaModel(model="llama3.1", temperature=0)
    relevancy_metrics = AnswerRelevancyMetric(threshold=0.7, model=judge_model)
    faithfulness_metrics = FaithfulnessMetric(threshold=0.7, model=judge_model)
    contextual_recall_metrics = ContextualRecallMetric(threshold=0.7, model=judge_model)
    contextual_precision_metrics = ContextualPrecisionMetric(threshold=0.7, model=judge_model)

    test_question = "What is the token expiration lifetime for the access token?"
    perfect_answer = "15 minutes."
    live_answer, live_contexts = run_rag_pipeline(test_question)
    print(f"Test Question: {test_question}")
    print(f"Perfect Answer: {perfect_answer}")
    print(f"Live Answer: {live_answer}")
    print(f"Live Contexts: {live_contexts}")

    test_case = LLMTestCase(
        input=test_question,
        expected_output=perfect_answer,
        actual_output=live_answer,
        retrieval_context=live_contexts,
    )

    results = evaluate(
        test_cases=[test_case],
        metrics=[
            relevancy_metrics,
            faithfulness_metrics,
            contextual_recall_metrics,
            contextual_precision_metrics,
        ],
    )
    print(results)


if __name__ == "__main__":
    try:
        run_baseline_evaluation()
    except Exception as ex:
        print(f"Something went wrong: {str(ex)}")
