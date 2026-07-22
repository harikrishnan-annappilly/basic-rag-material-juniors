from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def expand_user_query(original_query: str):
    print(f"Original query: {original_query}")

    llm = ChatOllama(model="llama3.1", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI search optimizer. Your goal is to rewrite the user's query into 3 distinct, highly targeted variations to improve vector database retrieval. Focus heavily on the core subject. Return ONLY the 3 variations, separated by newlines.",
            ),
            ("human", "{question}"),
        ]
    )

    # We build a simple LangChain pipeline
    chain = prompt | llm | StrOutputParser()

    expanded_queries_text = chain.invoke({"question": original_query})

    query_list = expanded_queries_text.strip().split("\n")

    print("Expanded Queries:")
    for i, q in enumerate(query_list):
        print(f"{i+1}. {q}")

    return query_list


if __name__ == "__main__":
    test_query = "What is the token expiration lifetime for the access token?"
    expand_user_query(test_query)
