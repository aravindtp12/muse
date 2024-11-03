from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def create_llm(model_name="llama3"):
    """Create an Ollama LLM instance with streaming capability."""
    # Initialize the Ollama LLM with streaming
    llm = Ollama(
        model=model_name,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.7,
    )
    return llm

def create_vectorstore(pdf_path):
    """Create a vector store from a PDF document."""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print("Splitting documents into chunks and creating embeddings")
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(
        documents=tqdm(splits, desc="Creating embeddings"),
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )
    print("Vector store created")
    
    return vectorstore

def create_qa_chain(vectorstore):
    """Create a question-answering chain with RAG capabilities."""
    llm = create_llm(model_name="llama3")
    print("Creating QA chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

def create_chain(prompt_template):
    """Create a LangChain chain with the given prompt template."""
    llm = create_llm(model_name="llama3")
    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def main():
    # Load PDF and create vector store
    pdf_path = "/Users/aravindmanoj/Downloads/the_vital_question_intro.pdf"
    vectorstore = create_vectorstore(pdf_path)
    
    # Create the QA chain
    qa_chain = create_qa_chain(vectorstore)
    
    # Interactive loop
    print("Ask questions about your PDF (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Generate response
        response = qa_chain({"query": user_input})
        print("\nAssistant:", response['result'])
        print("\nSources:", [doc.metadata for doc in response['source_documents']])


if __name__ == "__main__":
    main()