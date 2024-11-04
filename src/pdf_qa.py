from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
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
        temperature=0.1,
    )
    return llm

def create_vectorstore(pdf_path):
    """Create a vector store from a PDF document."""
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages from PDF")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200
    )
    
    # Splitting the documents into chunks
    texts = text_splitter.split_documents(documents=documents)
    
    print("\nCreating embeddings and vector store...")
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(
        documents=tqdm(texts),
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )
    
    return vectorstore

def create_qa_chain(vectorstore, llm, prompt_template):
    """Create a question-answering chain with RAG capabilities."""
    
    # prompt = PromptTemplate.from_template(prompt_template)
    print("Creating QA chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        # chain_type_kwargs={'prompt': prompt},
        return_source_documents=True
    )
    
    return qa_chain
