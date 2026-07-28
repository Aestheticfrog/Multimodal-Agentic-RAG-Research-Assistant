"""Streamlit Modern Dashboard for ResearchPilot AI."""
import os
import httpx
import streamlit as st

st.set_page_config(
    page_title="ResearchPilot AI – Agentic Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 700; color: #4A90E2; }
    .sub-caption { font-size: 1.05rem; color: #888888; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 6px; padding: 0 20px; }
    .citation-card { background-color: #1E222A; padding: 12px; border-radius: 8px; border-left: 4px solid #4A90E2; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 ResearchPilot AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Agentic Multimodal Research Assistant powered by <b>LangGraph</b> & <b>Google Gemini</b></div>', unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ System Control")
    
    # Backend Health Check
    try:
        res = httpx.get(f"{BACKEND_URL}/", timeout=2.0)
        if res.status_code == 200:
            st.success("🟢 Backend Service: Online")
        else:
            st.warning("🟡 Backend Service: Degraded")
    except Exception:
        st.error("🔴 Backend Service: Offline (Start `uvicorn backend.app.main:app --reload`)")

    st.divider()
    st.header("📚 Paper Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multi-page research papers to parse text & visual context into ChromaDB."
    )

    if uploaded_files:
        for file in uploaded_files:
            if st.button(f"📥 Process & Index '{file.name}'", key=file.name):
                with st.spinner(f"Parsing & indexing '{file.name}'..."):
                    try:
                        files = {"file": (file.name, file.getvalue(), "application/pdf")}
                        response = httpx.post(f"{BACKEND_URL}/api/v1/upload", files=files, timeout=300.0)
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"Indexed {data['chunks_indexed']} chunks successfully!")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Navigation Tabs
tab_chat, tab_lit_review, tab_about = st.tabs(["💬 Agent Research Chat", "📝 Literature Review Generator", "ℹ️ Architecture & Resume Info"])

with tab_chat:
    col_chat, col_citations = st.columns([2, 1])

    with col_chat:
        st.subheader("Interactive Research Assistant")
        
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User Query Input
        if prompt := st.chat_input("Ask a question about your uploaded research papers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("LangGraph agent is analyzing context, grading documents & generating response..."):
                    try:
                        response = httpx.post(
                            f"{BACKEND_URL}/api/v1/query",
                            json={"question": prompt},
                            timeout=300.0,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            answer = data["generation"]
                            citations = data.get("citations", [])
                            retry_count = data.get("retry_count", 0)

                            st.markdown(answer)
                            final_question = data.get("question")
                            orig_question = data.get("original_question", prompt)

                            with st.expander("🔍 LangGraph Agent Execution Info", expanded=(retry_count > 0)):
                                if retry_count > 0 or (final_question and final_question.strip() != orig_question.strip()):
                                    st.warning(f"🔄 **Query Autocorrected / Transformed** ({retry_count} transformation loops)")
                                    st.markdown(f"**Original Prompt:** `{orig_question}`")
                                    st.markdown(f"**Optimized Query:** `{final_question}`")
                                else:
                                    st.success("⚡ **Direct Match**: Retrieved relevant context on the first pass without needing query rewrite.")
                                    st.markdown(f"**Executed Query:** `{orig_question}`")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "citations": citations,
                            })
                            st.session_state["latest_citations"] = citations
                        else:
                            st.error(f"API Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    with col_citations:
        st.subheader("🏷️ Citation Breakdown")
        latest_citations = st.session_state.get("latest_citations", [])
        if latest_citations:
            for cit in latest_citations:
                st.markdown(f"""
                <div class="citation-card">
                    <b>Citation [{cit.get('id')}]</b><br/>
                    📄 <i>Source:</i> {cit.get('source')}<br/>
                    📌 <i>Page Reference:</i> {cit.get('page')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No citations active yet. Ask a question to see source groundings.")

with tab_lit_review:
    st.subheader("📝 Automated Literature Review Generator")
    st.write("Synthesize multi-paper research evidence into a structured synthesis report.")

    topic_input = st.text_input(
        "Literature Review Topic / Research Area:",
        value="Agentic RAG, Self-Correction, and Adaptive Query Routing in Modern LLM Systems"
    )

    if st.button("🚀 Generate Literature Review Report"):
        with st.spinner("Synthesizing theoretical foundations, methodologies, and open gaps..."):
            try:
                res = httpx.post(
                    f"{BACKEND_URL}/api/v1/literature-review",
                    json={"topic": topic_input},
                    timeout=300.0,
                )
                if res.status_code == 200:
                    review_data = res.json()
                    st.success("Literature Review Generated Successfully!")
                    st.markdown("### Executive Summary")
                    st.info(review_data["executive_summary"])
                    st.markdown("---")
                    st.markdown(review_data["full_report"])
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to generate review: {e}")

with tab_about:
    st.subheader("🏗️ Architecture & Resume Highlights")
    st.markdown("""
    ### Why ResearchPilot AI Stands Out
    - **LangGraph Controlled State Workflow**: Unlike basic linear RAG chains, ResearchPilot AI implements an agentic graph with dynamic decision nodes (`retrieve`, `grade_documents`, `transform_query`, `generate_answer`, `hallucination_check`).
    - **Google Gemini Integration**: Uses `gemini-2.5-flash` for high-speed structured JSON grading and multimodal context comprehension.
    - **Decoupled FastAPI Architecture**: Clean REST endpoints serving an interactive Streamlit UI.
    
    ### Tech Stack
    - **Agents**: LangGraph, LangChain
    - **LLM Engine**: Google Gemini
    - **Vector Storage**: ChromaDB
    - **Backend API**: FastAPI, Uvicorn, Pydantic
    - **Frontend UI**: Streamlit
    - **PDF Processing**: PyMuPDF (`fitz`)
    """)
