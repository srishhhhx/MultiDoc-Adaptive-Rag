"""
Utility functions for the Advanced RAG application
"""
import shutil
import os
import streamlit as st
from config import CHROMA_PERSIST_DIR


def clear_chroma_db():
    """Clear ChromaDB data directory for fresh start"""
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("Cleared existing ChromaDB data for fresh start")


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'graph_instance' not in st.session_state:
        st.session_state.graph_instance = None
    if 'db_cleared' not in st.session_state:
        st.session_state.db_cleared = False


def get_file_key(uploaded_file):
    """Generate unique key for uploaded file"""
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}_{uploaded_file.size}"


def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes >= 1024 * 1024:
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    elif size_bytes >= 1024:
        size_kb = size_bytes / 1024
        return f"{size_kb:.1f} KB"
    else:
        return f"{size_bytes} bytes"
