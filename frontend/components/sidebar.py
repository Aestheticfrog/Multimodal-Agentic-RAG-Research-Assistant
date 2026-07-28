"""Sidebar component for Streamlit dashboard."""
import streamlit as st


def render_sidebar():
    """Render application sidebar."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.text_input("API Key Status", value="Configured via .env", disabled=True)
        st.divider()
        st.header("📚 Document Library")
        st.info("No documents uploaded yet.")
