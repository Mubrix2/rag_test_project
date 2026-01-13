from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_documents(data_path="data/"):
    # 1. Load all PDFs in the data folder
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    
    # 2. Define the Splitter
    # chunk_size: characters per piece. chunk_overlap: keeps context between pieces.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    # 3. Create the chunks
    docs = text_splitter.split_documents(raw_docs)
    print(f"Loaded {len(raw_docs)} pages and split into {len(docs)} chunks.")
    return docs