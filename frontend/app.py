import streamlit as st
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DevSynt RAG Chatbot Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DevSynt AI Document RAG Dashboard")

# Fetch stats from backend
try:
    stats_res = requests.get(f"{API_URL}/stats").json()
    total_docs = stats_res.get("total_documents", 0)
    total_chunks = stats_res.get("total_chunks", 0)
    doc_list = stats_res.get("documents", [])
except Exception:
    total_docs, total_chunks, doc_list = 0, 0, []

# Initialize Chat Session Stats
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

# --- Top Metrics Panel ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Documents", total_docs)
col2.metric("Indexed Chunks", total_chunks)
col3.metric("Total Queries", st.session_state.chat_count)
col4.metric("System Status", "Active" if total_chunks > 0 else "Ready")

st.markdown("---")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📄 Document Management")
    
    # New Chat Button
    if st.button("➕ New Chat Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    # Upload Section
    uploaded_file = st.file_uploader("Upload Document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Extracting, cleaning & chunking..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(f"Indexed successfully! Added {res.json().get('chunks_added')} chunks.")
                    st.rerun()
                else:
                    st.error("Document upload/processing failed.")

    st.markdown("---")
    st.subheader("Indexed Documents")
    if doc_list:
        for doc in doc_list:
            st.text(f"✅ {doc}")
    else:
        st.info("No external uploads yet.")

# --- Main Chatbot Section ---
st.header("💬 Real Estate RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history with timestamps
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "timestamp" in msg:
            st.caption(f"🕒 {msg['timestamp']}")
        if "sources" in msg and msg["sources"]:
            st.info("📍 **Sources:** " + ", ".join(msg["sources"]))

# Input Handling
if user_query := st.chat_input("Ask a question about your documents..."):
    now_time = datetime.now().strftime("%I:%M %p")
    
    # Append User Query
    st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": now_time})
    with st.chat_message("user"):
        st.markdown(user_query)
        st.caption(f"🕒 {now_time}")

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating answer..."):
            try:
                res = requests.post(f"{API_URL}/chat", json={"question": user_query})
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.info("📍 **Sources:** " + ", ".join(sources))
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    st.session_state.chat_count += 1
                else:
                    st.error("Error communicating with RAG server.")
            except Exception as e:
                st.error(f"Backend Connection Error: {str(e)}")