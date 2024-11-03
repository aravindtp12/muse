from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_llm(model_name="llama2"):
    """Create an Ollama LLM instance with streaming capability."""
    # Initialize the Ollama LLM with streaming
    llm = Ollama(
        model=model_name,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.7,
    )
    return llm

def create_chain(prompt_template):
    """Create a LangChain chain with the given prompt template."""
    llm = create_llm(model_name="llama3")
    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def main():
    # Example prompt template
    prompt_template = """
    You are a helpful assistant. Please respond to the following question:
    Question: {question}
    Answer:"""
    
    # Create the chain
    chain = create_chain(prompt_template)
    
    # Interactive loop
    print("Chat with the LLama model (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Generate response
        response = chain.invoke({"question": user_input})
        print("\nAssistant:", response['text'])

if __name__ == "__main__":
    main()