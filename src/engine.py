from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_rag_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2:1b")
    
    # Notice the prompt MUST have the {context} and {input} variables
    system_prompt = (
        "You are a helpful assistant. Use the context below to answer the question. "
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 1. Create the 'Answer' logic
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # 2. Create the Retrieval Chain
    # We pass the vectorstore as a retriever
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(), 
        combine_docs_chain
    )
    
    return retrieval_chain