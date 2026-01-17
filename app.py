import streamlit as st
import os
from src.ingestion import load_and_split_documents
from src.database import get_vectorstore
from src.engine import get_rag_chain

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local AI Assistant", page_icon="🤖")
st.title("📄 Private Document RAG")
st.markdown("Ask questions about your local PDFs. All processing is done on your CPU.")

# --- SIDEBAR: INITIALIZATION ---
with st.sidebar:
    st.header("Setup")
    if st.button("🔄 Re-index Documents"):
        with st.spinner("Reading PDFs..."):
            chunks = load_and_split_documents()
            st.session_state.vectorstore = get_vectorstore(chunks)
            st.success("Database Updated!")

# --- INITIALIZE SESSION STATE ---
# We store the chain in 'session_state' so it doesn't reload every time you click a button
if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = get_vectorstore()
    except:
        st.warning("Please put PDFs in the 'data' folder and click Re-index.")
        st.stop()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_rag_chain(st.session_state.vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CHAT INTERFACE ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
# Inside your Chat Input section in app.py
if prompt := st.chat_input("What is in my documents?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Format the chat history for LangChain
        # We convert our message list into a format the chain expects
        chat_history = []
        for m in st.session_state.messages[:-1]: # everything except the current prompt
             chat_history.append((m["role"], m["content"]))

        response = st.session_state.rag_chain.invoke({
            "input": prompt,
            "chat_history": chat_history # Passing the memory here!
        })
        
        answer = response["answer"]
        st.markdown(answer)
        # ... (rest of your source tracking code)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # NEW for LangGraph:
            initial_state = {"question": prompt}
            output_state = st.session_state.rag_chain.invoke(initial_state)

            answer = output_state["generation"]
            st.markdown(answer)

            with st.expander("Agentic Metadata"):
                st.write(f"Documents Retrieved: {len(output_state['documents'])}")
                for doc in output_state['documents']:
                    st.write(f"- {doc.metadata.get('source')}")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})