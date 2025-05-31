import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv
from groq import Groq

from rag_chain2 import (
    build_rag_chain_from_files,
    process_scheme_query
)

load_dotenv()

st.set_page_config(page_title="CMRF RAG Minimal", layout="centered")
st.title("🤖 CMRF RAG Minimal Assistant")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file")
    st.stop()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Upload Files
pdf_file = st.file_uploader("Upload Scheme PDF", type=["pdf"])
txt_file = st.file_uploader("Upload Scheme TXT", type=["txt"])

if not (pdf_file or txt_file):
    st.warning("Upload at least one document to proceed.")
    st.stop()

# Build RAG Chain
if st.session_state.rag_chain is None:
    with st.spinner("🔧 Building RAG system..."):
        try:
            st.session_state.rag_chain = build_rag_chain_from_files(
                pdf_file, txt_file, GROQ_API_KEY
            )
            st.success("✅ RAG system ready")
        except Exception as e:
            st.error(f"Failed to build RAG system: {e}")
            st.stop()

# Query Input
query = st.text_input("Ask a question about schemes", placeholder="e.g. Show main schemes")
if st.button("🔍 Get Answer") and query:
    with st.spinner("Processing query..."):
        response = process_scheme_query(st.session_state.rag_chain, query)

        # Ensure output formatting for detailed response structure
        st.markdown("### 📋 Detailed Answer:")

        if "schemes:" in response.lower():
            st.success(response)
        else:
            fallback = "\n\nI don't have relevant information for this. You can contact 102/104 helpline numbers for more details."
            st.warning(response + fallback)
