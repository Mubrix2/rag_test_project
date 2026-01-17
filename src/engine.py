from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_rag_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2:1b")
    retriever = vectorstore.as_retriever()

    # 1. Contextualize Question: This re-writes the user's question to be independent
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    # This creates a "Smart Retriever" that understands history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer Question: This handles the final response
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # The final "Brain" of your bot
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)