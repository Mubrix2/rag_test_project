import os
from src.ingestion import load_and_split_documents
from src.database import get_vectorstore
from src.engine import get_rag_chain

def run_app():
    print("--- Starting Local RAG Assistant ---")

    # 1. SETUP DATA & DATABASE
    # We check if data exists; if not, we try to load an existing database
    if os.path.exists("data") and any(f.endswith('.pdf') for f in os.listdir("data")):
        try:
            chunks = load_and_split_documents()
            vectorstore = get_vectorstore(chunks)
        except Exception as e:
            print(f"Error during ingestion: {e}. Trying to load existing DB...")
            vectorstore = get_vectorstore()
    else:
        # If data folder is empty, just try to load whatever was saved before
        vectorstore = get_vectorstore()

    # 2. INITIALIZE THE CHAIN
    rag_chain = get_rag_chain(vectorstore)

    # 3. CHAT LOOP
    print("\nSystem Ready! Type your questions below.")
    while True:
        query = input("\nUser: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        # This is the line you asked about!
        # It sends a dictionary to the engine
        response = rag_chain.invoke({"input": query})
        
        # 4. PRINT RESPONSE & SOURCES
        print(f"\nAI: {response['answer']}")
        
        print("\nSources Used:")
        # We loop through the 'context' to see which files the AI read
        source_docs = response.get("context", [])
        seen_sources = set()
        
        for doc in source_docs:
            meta = doc.metadata
            source_info = f"- {os.path.basename(meta.get('source', 'Unknown'))} (Page {meta.get('page', 0) + 1})"
            if source_info not in seen_sources:
                print(source_info)
                seen_sources.add(source_info)

if __name__ == "__main__":
    run_app()