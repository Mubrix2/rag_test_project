from typing import List, TypedDict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document

# 1. DEFINE THE STATE
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# 2. DEFINE THE NODES
def retrieve(state: GraphState, retriever):
    print("---ACTION: RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    print("---ACTION: GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    llm = OllamaLLM(model="llama3.2:1b")
    
    # Prompt
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Chain
    chain = prompt | llm
    
    # Run
    context = "\n\n".join([doc.page_content for doc in documents])
    response = chain.invoke({"context": context, "question": question})
    
    return {"generation": response, "documents": documents}

# 3. BUILD THE GRAPH
def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    
    workflow = StateGraph(GraphState)

    # Define the nodes - we use a lambda to pass the retriever to the retrieve function
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("generate", generate)

    # Build edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()