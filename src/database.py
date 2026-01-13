import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "faiss_index"

def get_vectorstore(chunks=None):
    """Creates a new vector store or loads an existing one from the disk."""
    # Use a lightweight model perfect for Windows CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Case A: If chunks are provided, we create a NEW database
    if chunks:
        print("Indexing documents into FAISS...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(DB_PATH)
        return vectorstore
    
    # Case B: If no chunks, try to load an existing one
    if os.path.exists(DB_PATH):
        print("Loading existing index from disk...")
        # allow_dangerous_deserialization is required for local loading
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    raise ValueError("No existing index found. Please provide chunks to create one.")