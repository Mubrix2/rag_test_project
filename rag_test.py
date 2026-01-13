import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- 1. THE KNOWLEDGE BASE ---
documents = [
    "The company policy says employees get 20 days of vacation per year.",
    "The office coffee machine is broken and will be fixed on Friday.",
    "The password for the guest WiFi is 'BlueLlama2024'.",
    "New employees must complete their security training within 3 days.",
    "Mubarak likes drinking tea a lot",
    "The only way to take revenge is to put in your best"
]

# --- 2. RETRIEVAL SYSTEM (From Part 1) ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --- 3. THE USER QUERY ---
query = "Is there a way to take revenge?"

# Find the best document
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=1)
retrieved_context = documents[I[0][0]]

print(f"--- Search found: {retrieved_context} ---\n")

# --- 4. GENERATION (Asking Ollama to answer using that context) ---
prompt = f"""
You are a helpful office assistant. 
Use the following piece of context to answer the user's question.
If the answer is not in the context, say you don't know.

Context: {retrieved_context}
Question: {query}
"""

response = ollama.generate(model='llama3.2:1b', prompt=prompt)

print("--- AI Response ---")
print(response['response'])