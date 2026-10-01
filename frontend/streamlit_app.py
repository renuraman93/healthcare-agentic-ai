"""
Streamlit frontend for the Healthcare Agentic AI Chatbot.

All business logic lives in the FastAPI backend; this file only
renders UI state and calls api_client.
"""

import os
import sys

import streamlit as st

# Allow running `streamlit run frontend/streamlit_app.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.api_client import ApiClient, ApiError

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")

# ---- session state ----
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-" + os.urandom(4).hex()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"|"assistant", "content", "sources", "agent"}]

client = ApiClient(API_BASE_URL)

st.title("🏥 Healthcare AI Assistant")
st.caption(
    "Educational demo only — not medical advice. Does not diagnose and "
    "cannot replace a healthcare professional."
)

# ---- sidebar: connection status + document upload ----
with st.sidebar:
    st.subheader("Backend status")
    try:
        health = client.health()
        st.success(f"Connected — {health['vector_store_chunks']} chunks indexed")
    except ApiError as e:
        st.error(str(e))
        st.stop()

    st.divider()
    st.subheader("Upload Medical Document")
    uploaded_file = st.file_uploader(
        "PDF, TXT, or DOCX", type=["pdf", "txt", "docx"]
    )
    if uploaded_file is not None and st.button("Upload and Index", use_container_width=True):
        with st.spinner("Uploading..."):
            try:
                upload_result = client.upload_document(
                    uploaded_file.name, uploaded_file.getvalue()
                )
                st.info(f"Uploaded: {upload_result['filename']}")
            except ApiError as e:
                st.error(f"Upload failed: {e}")
                upload_result = None

        if upload_result:
            with st.spinner("Indexing..."):
                try:
                    index_result = client.index_document(upload_result["filename"])
                    st.success(
                        f"Indexed {index_result['chunks_indexed']} chunks "
                        f"from {index_result['filename']}"
                    )
                except ApiError as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.subheader("Indexed Documents")
    try:
        docs = client.list_documents()
        if docs["documents"]:
            for d in docs["documents"]:
                st.text(f"📄 {d['source']}")
            st.caption(f"Total chunks: {docs['total_chunks']}")
        else:
            st.caption("No documents indexed yet.")
    except ApiError as e:
        st.error(str(e))

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = "streamlit-" + os.urandom(4).hex()
        st.rerun()

# ---- main chat area ----
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.text(f"- {s['source']} (chunk {s['chunk_index']})")

user_message = st.chat_input("Ask about your medical documents or general health questions...")

if user_message:
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = client.chat(user_message, st.session_state.session_id)
                answer = result["answer"]
                sources = result.get("sources", [])
                if result.get("emergency_flag"):
                    st.warning("⚠️ Emergency wording detected in this message.")
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.text(f"- {s['source']} (chunk {s['chunk_index']})")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources, "agent": result.get("agent")}
                )
            except ApiError as e:
                st.error(f"Error: {e}")