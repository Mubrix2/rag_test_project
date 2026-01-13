from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# 1. LOAD A PDF (You can use any PDF file you have on your PC)
# Replace 'your_file.pdf' with a path to a real PDF
# For now, let's pretend we have a text-based input to keep it simple:
from langchain_core.documents import Document
# raw_documents = [
#     Document(page_content="The company's health insurance covers dental, vision, and mental health. However, it does not cover cosmetic surgery. Claims must be submitted within 30 days of the visit.", metadata={"source": "hr-manual"}),
#     Document(page_content="The office is open from 9 AM to 6 PM. Employees can work from home on Wednesdays. Parking is free for all full-time staff in the basement level.", metadata={"source": "hr-manual"})
# ]


loader = PyPDFLoader("Complete Draft.pdf")
raw_documents = loader.load()

# 2. CHUNKING (Splitting the text into manageable pieces)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(raw_documents)

# 3. EMBEDDINGS & VECTOR STORE (The "Search Engine")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# 4. SETUP OLLAMA
llm = OllamaLLM(model="llama3.2:1b")

# 5. THE "CHAIN" (Connecting everything)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# 6. ASK A QUESTION
query = "Can you name the Team?"
response = qa_chain.invoke(query)

print("\n--- AI ANSWER ---")
print(response["result"])