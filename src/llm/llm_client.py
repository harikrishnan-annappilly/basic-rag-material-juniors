from langchain_ollama import ChatOllama


def get_llm():
    """
    Initializes and returns the local Ollama LLM client.
    """
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    return llm


if __name__ == "__main__":
    try:
        print("Initializing local LLM model")
        llm_client = get_llm()
        print("Sending a test message to Llama 3.2")
        response = llm_client.invoke("Say 'Hello World!' if you can hear me!")
        print("--- AI Response ---")
        print(response.text)
    except Exception as ex:
        print(f"Something went wrong {ex}")
