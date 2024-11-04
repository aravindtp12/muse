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
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages from PDF")
    
    # Process pages in order and add page numbers to help track order
    texts_with_metadata = []
    for page_num, doc in enumerate(documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(doc.page_content)
        print(f"Page {page_num + 1}: Created {len(chunks)} chunks")
        
        # Verify content (print first 100 chars of first chunk)
        if chunks:
            print(f"Sample from page {page_num + 1}: {chunks[0][:100]}...")
        
        # Add page number and chunk index as metadata
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) > 0:  # Only add non-empty chunks
                texts_with_metadata.append({
                    "text": chunk,
                    "metadata": {
                        "page": page_num + 1,
                        "chunk": chunk_idx + 1,
                        "source": pdf_path
                    }
                })
    
    print(f"\nTotal chunks created: {len(texts_with_metadata)}")
    
    if not texts_with_metadata:
        raise ValueError("No text chunks were created! Check if the PDF is readable.")
    
    print("\nCreating embeddings and vector store...")
    embeddings = OllamaEmbeddings(model="llama3")
    
    # Create vector store with metadata
    texts = [t["text"] for t in texts_with_metadata]
    metadatas = [t["metadata"] for t in texts_with_metadata]
    
    # Verify we're sending non-empty data to Chroma
    print(f"Number of texts being sent to vector store: {len(texts)}")
    print(f"First text sample: {texts[0][:100]}...")
    print(f"Last text sample: {texts[-1][:100]}...")
    
    vectorstore = Chroma.from_texts(
        texts=tqdm(texts),
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )
    
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