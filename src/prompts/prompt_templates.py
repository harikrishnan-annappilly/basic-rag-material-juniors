from langchain_core.prompts import PromptTemplate


def get_template():
    """
    This function creates and returns our standard RAG prompt templates
    """
    template_string = """
You are a helpful and polite assistant.
Answer the user's question using ONLY the provided context below.
If the answer is not contained in the context, say "I don't know based on the provided document."
The answer should be only one sentence long.

Context:
{context}

Question:
{question}

Answer:"""
    prompt_template = PromptTemplate.from_template(template_string)
    return prompt_template


if __name__ == "__main__":
    try:
        my_prompt = get_template()
        fake_context = "The company's Q3 revenue was 4.5 million dollars."
        fake_question = "How much money did the company make in Q3?"
        formatted_prompt = my_prompt.format(context=fake_context, question=fake_question)
        print("✅ Successfully formatted the prompt!\n")
        print("--- What the AI will actually see ---")
        print(formatted_prompt)
    except Exception as ex:
        print(f"Something went wrong {ex}")
