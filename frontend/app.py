"""Streamlit Modern Dashboard for ResearchPilot AI with standalone fallback & live resume analytics."""
import os
import sys
from pathlib import Path
import httpx
import streamlit as st

# Add project root directory to sys.path for Streamlit Cloud deployment
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

st.set_page_config(
    page_title="Agentic Multimodal RAG – Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ─── Fast Backend Availability Check (once per session) ───
def _is_backend_online():
    if "backend_online" not in st.session_state:
        try:
            res = httpx.get(f"{BACKEND_URL}/", timeout=0.5)
            st.session_state["backend_online"] = (res.status_code == 200)
        except Exception:
            st.session_state["backend_online"] = False
    return st.session_state["backend_online"]

def should_use_backend():
    """Determines whether to use FastAPI REST API or Standalone Direct Engine.
    Respects user mode selection (Auto-Detect, Force Backend, Force Standalone).
    """
    mode = st.session_state.get("execution_mode", "Auto-Detect (Smart Routing)")
    if mode == "🌐 Force FastAPI REST API":
        return True
    elif mode == "⚡ Force Standalone Direct Engine":
        return False
    # Default: Auto-Detect
    return _is_backend_online()

# ─── Persist Dir for ChromaDB ───
PERSIST_DIR_ABS = str(root_dir / "database" / "chroma_store")

# Custom CSS styling
st.markdown("""
<style>
.main-title {font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 0.5rem 0;}
.sub-caption {font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">🔬 ResearchPilot AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Self-Correcting Research Assistant powered by <b>LangGraph</b>, <b>Google Gemini</b> & <b>ChromaDB</b></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Helper: Parse & store PDF in BOTH session state and ChromaDB
# ═══════════════════════════════════════════════════════════
def process_pdf_upload(file):
    if should_use_backend():
        try:
            files = {"file": (file.name, file.getvalue(), "application/pdf")}
            response = httpx.post(f"{BACKEND_URL}/api/v1/upload", files=files, timeout=30.0)
            if response.status_code == 200:
                return response.json()["chunks_indexed"]
        except Exception:
            pass

    # Direct standalone mode (Streamlit Cloud)
    from backend.app.utils.pdf_parser import parse_pdf_bytes
    from backend.app.retrievers.vector_store import add_documents_to_vector_store

    pdf_bytes = file.getvalue()
    docs = parse_pdf_bytes(pdf_bytes, file.name)
    if not docs:
        return 0

    # Store RAW PDF BYTES in session state — the MOST bulletproof source of truth.
    # Document objects may fail to serialize across Streamlit reruns.
    # Raw bytes are just bytes — they ALWAYS serialize correctly.
    if "pdf_bytes_store" not in st.session_state:
        st.session_state["pdf_bytes_store"] = {}
    st.session_state["pdf_bytes_store"][file.name] = pdf_bytes

    # Also store parsed docs (belt AND suspenders)
    if "in_memory_docs" not in st.session_state:
        st.session_state["in_memory_docs"] = []
    st.session_state["in_memory_docs"] = [
        d for d in st.session_state["in_memory_docs"]
        if d.metadata.get("source") != file.name
    ] + docs

    # Also store in ChromaDB (best-effort, not required)
    try:
        add_documents_to_vector_store(docs)
    except Exception:
        pass

    return len(docs)


# ═══════════════════════════════════════════════════════════
# Helper: Execute agent query — re-parses PDFs from raw bytes every time
# ═══════════════════════════════════════════════════════════
def execute_agent_query(prompt_text):
    from backend.app.utils.security import moderate_query
    is_safe, refusal_reason = moderate_query(prompt_text)
    if not is_safe:
        return {
            "question": prompt_text,
            "original_question": prompt_text,
            "generation": refusal_reason,
            "citations": [],
            "sources": [],
            "retry_count": 0,
        }

    if should_use_backend():
        try:
            response = httpx.post(
                f"{BACKEND_URL}/api/v1/query",
                json={"question": prompt_text},
                timeout=60.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

    # ── Bulletproof document reconstruction from raw PDF bytes ──
    # This re-parses from stored bytes on EVERY query.
    # Raw bytes in session state are guaranteed to survive Streamlit reruns.
    from backend.app.utils.pdf_parser import parse_pdf_bytes
    from backend.app.agents.graph import researchpilot_agent

    fresh_docs = []
    pdf_store = st.session_state.get("pdf_bytes_store", {})
    for filename, pdf_bytes in pdf_store.items():
        try:
            parsed = parse_pdf_bytes(pdf_bytes, filename)
            fresh_docs.extend(parsed)
        except Exception:
            pass

    # Fallback to in_memory_docs if pdf_bytes_store is empty
    if not fresh_docs:
        fresh_docs = list(st.session_state.get("in_memory_docs", []))

    initial_state = {
        "question": prompt_text,
        "original_question": prompt_text,
        "documents": fresh_docs,
        "generation": "",
        "web_search_needed": False,
        "hallucination_grade": "",
        "answer_grade": "",
        "retry_count": 0,
        "citations": [],
    }
    res = researchpilot_agent.invoke(initial_state)
    citations = res.get("citations", [])
    unique_sources = sorted(list(set([c.get("source") for c in citations if c.get("source")])))
    return {
        "question": res.get("question", prompt_text),
        "original_question": prompt_text,
        "generation": res.get("generation", ""),
        "citations": citations,
        "sources": unique_sources,
        "retry_count": res.get("retry_count", 0),
    }


def execute_literature_review(topic_text):
    if should_use_backend():
        try:
            res = httpx.post(
                f"{BACKEND_URL}/api/v1/literature-review",
                json={"topic": topic_text},
                timeout=300.0,
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("full_report") or data.get("review") or ""
        except Exception:
            pass

    # Standalone direct generation fallback (Streamlit Cloud)
    try:
        from backend.app.models.llm import get_gemini_llm
        from backend.app.utils.pdf_parser import parse_pdf_bytes
        from backend.app.agents.nodes import extract_text_content

        fresh_docs = []
        pdf_store = st.session_state.get("pdf_bytes_store", {})
        for filename, pdf_bytes in pdf_store.items():
            try:
                parsed = parse_pdf_bytes(pdf_bytes, filename)
                fresh_docs.extend(parsed)
            except Exception:
                pass

        if not fresh_docs:
            fresh_docs = list(st.session_state.get("in_memory_docs", []))

        formatted_context = "\n---\n".join([d.page_content for d in fresh_docs[:15]]) if fresh_docs else "No specific PDF documents uploaded in active session."

        llm = get_gemini_llm(temperature=0.3)
        prompt = f"""You are a senior research professor writing a Literature Review Synthesis Report on the topic: '{topic_text}'.

Available Research Evidence in Active Session:
{formatted_context}

Write a high-quality, structured Literature Review report in Markdown format including:
1. Executive Summary
2. Core Theoretical Foundations & Taxonomy
3. Comparative Analysis of Approaches / Methodologies
4. Key Research Gaps & Open Challenges
5. Future Directions
"""
        res = llm.invoke(prompt)
        raw_report = res.content if hasattr(res, "content") else str(res)
        return extract_text_content(raw_report)
    except Exception as e:
        return f"Error generating literature review: {str(e)}"


def get_library_summary():
    try:
        from backend.app.retrievers.vector_store import get_indexed_sources_summary
        return get_indexed_sources_summary()
    except Exception:
        return {}


def reset_library():
    try:
        import shutil
        if os.path.exists(PERSIST_DIR_ABS):
            shutil.rmtree(PERSIST_DIR_ABS)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# Sidebar Setup
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ System Configuration")
    mode = st.selectbox(
        "Execution Mode",
        ["Auto-Detect (Smart Routing)", "🌐 Force FastAPI REST API", "⚡ Force Standalone Direct Engine"],
        key="execution_mode",
        help="Auto-Detect automatically connects to your local FastAPI server on port 8000 when running locally, and switches to Standalone Direct Engine when deployed on Streamlit Cloud."
    )

    if should_use_backend():
        st.success("🟢 REST API Backend: Connected (http://localhost:8000)")
    else:
        st.info("⚡ Mode: Standalone Direct Agent Engine (Streamlit Cloud Active)")

    # Show session document count
    mem_docs = st.session_state.get("in_memory_docs", [])
    if mem_docs:
        mem_sources = set(d.metadata.get("source") for d in mem_docs)
        st.success(f"📄 **{len(mem_docs)} chunks** from **{len(mem_sources)} paper(s)** in memory")
    else:
        st.warning("📄 No papers in memory. Upload PDFs below.")

    st.divider()
    st.header("📚 Paper Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multi-page research papers to parse text & visual context into ChromaDB."
    )

    if uploaded_files:
        current_file_names = sorted([f.name for f in uploaded_files])
        last_file_names = st.session_state.get("last_uploaded_names", [])

        st.caption(f"📁 Selected for Indexing: **{len(uploaded_files)} PDF file(s)**")
        for f in uploaded_files:
            st.text(f"• {f.name}")

        process_clicked = st.button("🚀 Process & Index All Uploaded Papers", type="primary", use_container_width=True)
        should_process = (current_file_names != last_file_names) or process_clicked

        if should_process:
            # Clear stale data
            st.session_state["in_memory_docs"] = []
            st.session_state["pdf_bytes_store"] = {}
            try:
                from backend.app.retrievers.vector_store import clear_vector_store
                clear_vector_store()
            except Exception:
                pass

            total_added = 0
            for file in uploaded_files:
                with st.spinner(f"Parsing & indexing '{file.name}'..."):
                    count = process_pdf_upload(file)
                    total_added += count
                    if count == 0:
                        st.warning(f"⚠️ '{file.name}' returned 0 text chunks. Check if the PDF is scanned or image-only.")
                    else:
                        st.info(f"✅ '{file.name}' indexed: {count} chunks")

            st.session_state.chunks_indexed = total_added
            st.session_state["last_uploaded_names"] = current_file_names
            st.success(f"Indexed {total_added} total chunks across {len(uploaded_files)} paper(s) successfully!")


# Initialize session defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks_indexed" not in st.session_state:
    st.session_state.chunks_indexed = 0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ═══════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════
tab_chat, tab_lit_review, tab_metrics = st.tabs(["💬 Agent Research Chat", "📝 Literature Review Generator", "📊 Live Resume Metrics & Architecture"])

with tab_chat:
    col_chat, col_citations = st.columns([2, 1])

    with col_chat:
        st.subheader("Interactive Agentic Assistant")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "execution_info" in message:
                    info = message["execution_info"]
                    retry_count = info.get("retry_count", 0)
                    final_q = info.get("final_question", "")
                    orig_q = info.get("orig_question", "")
                    with st.expander("🔍 LangGraph Agent Execution Info", expanded=True):
                        sources = info.get("sources", [])
                        if sources:
                            st.info(f"📚 **Indexed Paper Sources in Memory:** `{', '.join(sources)}` ({len(sources)} paper(s) active)")
                        if retry_count > 0 or (final_q and final_q.strip() != orig_q.strip()):
                            st.warning(f"🔄 **Query Autocorrected / Transformed** ({retry_count} transformation loops)")
                            st.markdown(f"**Original Prompt:** `{orig_q}`")
                            st.markdown(f"**Optimized Query:** `{final_q}`")
                        else:
                            st.success("⚡ **Direct Match**: Retrieved relevant context on the first pass without needing query rewrite.")
                            st.markdown(f"**Executed Query:** `{orig_q}`")

        if prompt := st.chat_input("Ask a question about your uploaded research papers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("LangGraph agent analyzing context, grading relevance & checking hallucinations..."):
                data = execute_agent_query(prompt)
                if data:
                    st.session_state.query_count += 1
                    answer = data["generation"]
                    citations = data.get("citations", [])
                    sources = data.get("sources", [])
                    retry_count = data.get("retry_count", 0)
                    final_question = data.get("question")
                    orig_question = data.get("original_question", prompt)

                    msg_data = {
                        "role": "assistant",
                        "content": answer,
                    }

                    if "🔒 Security Guardrail" not in answer:
                        msg_data["execution_info"] = {
                            "retry_count": retry_count,
                            "final_question": final_question,
                            "orig_question": orig_question,
                            "sources": sources,
                        }

                    st.session_state.messages.append(msg_data)
                    st.rerun()

    with col_citations:
        st.subheader("📎 Source Context & Citations")
        if st.session_state.messages:
            last_assistant = [m for m in st.session_state.messages if m["role"] == "assistant"]
            if last_assistant:
                info = last_assistant[-1].get("execution_info", {})
                sources_list = info.get("sources", [])
                if sources_list:
                    for s in sources_list:
                        st.markdown(f"- 📄 **{s}**")
                else:
                    st.info("No citations available for last response.")
        else:
            st.info("Ask a question to see citations here.")


with tab_lit_review:
    st.subheader("📝 Automated Literature Review Generator")
    topic = st.text_input("Enter a research topic to generate a literature review:")
    if st.button("Generate Literature Review"):
        if topic:
            with st.spinner("Generating comprehensive literature review..."):
                review = execute_literature_review(topic)
                st.markdown(review)
        else:
            st.warning("Please enter a research topic first.")


with tab_metrics:
    st.subheader("📊 Live Resume Metrics & Architecture")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    mem_docs_count = len(st.session_state.get("in_memory_docs", []))
    mem_sources_count = len(set(d.metadata.get("source") for d in st.session_state.get("in_memory_docs", [])))
    col_m1.metric("PDF Chunks in Memory", mem_docs_count)
    col_m2.metric("Papers in Memory", mem_sources_count)
    col_m3.metric("Agent Queries Executed", st.session_state.query_count)
    col_m4.metric("Backend Mode", "REST API" if should_use_backend() else "Standalone Engine")

    st.divider()
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
    │  Streamlit   │────▶│  LangGraph Agent │────▶│  Google Gemini│
    │  Frontend    │     │  State Machine   │     │  LLM Pool     │
    └─────────────┘     └────────┬─────────┘     └───────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌───▼───┐ ┌──────▼──────┐
              │  Retrieve  │ │ Grade │ │  Generate   │
              │  Documents │ │ Docs  │ │  Answer     │
              └─────┬──────┘ └───────┘ └─────────────┘
                    │
              ┌─────▼──────┐
              │  ChromaDB   │
              │  + Session  │
              │  Memory     │
              └─────────────┘
    ```
    """)

    st.divider()
    st.markdown("### 📋 In-Memory Document Details")
    mem_docs_detail = st.session_state.get("in_memory_docs", [])
    if mem_docs_detail:
        source_page_map = {}
        for d in mem_docs_detail:
            src = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "?")
            source_page_map.setdefault(src, []).append(page)
        for src, pages in source_page_map.items():
            st.markdown(f"**{src}**: {len(pages)} chunks across pages {sorted(set(pages))}")
    else:
        st.info("No papers in memory. Upload PDFs in the sidebar.")
