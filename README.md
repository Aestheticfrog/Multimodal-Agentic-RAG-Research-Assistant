# 🔬 Multimodal Agentic RAG Research Assistant (ResearchPilot AI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-agentic-rag-research-assistant-dsmrukzu4revjzyvuqor.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

**ResearchPilot AI** is an enterprise-grade, self-correcting RAG research assistant built to analyze complex academic papers, perform literature review synthesis, and provide grounded answers with exact source page citations.

Powered by **LangGraph**, **Google Gemini**, **ChromaDB**, **FastAPI**, and **Streamlit**.

---

## 🌟 Key Features & Innovations

- 🌐 **Dynamic Adaptive Retrieval & Source Stratification**: Classifies query intent automatically. Fact-based queries trigger targeted high-precision search ($k=6$), while comparative queries trigger deep multi-paper candidate retrieval ($k=20$) with balanced round-robin sampling across all uploaded PDF sources.
- 🎯 **Target Section Chunk Ranking & Keyword Match**: Pinpoints specific section queries (e.g., *Section 4.3*, *Table 1*, *Figure 2*) using exact regex token matching and automatically boosts target section chunks to the top of the LLM context window.
- 💾 **Zero-Drop Raw PDF Bytes Persistence**: Resolves Streamlit Cloud session state serialization issues by storing raw PDF binary streams directly in memory and re-parsing dynamically per query for 100% document retrieval reliability.
- 🛡️ **Resilient Failover Model Pool (`ResilientGeminiLLM`)**: Solves free-tier API rate limits (HTTP 429) by dynamically rotating requests across model candidates (`gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-3-flash`) in <50ms.
- 🔒 **Context-Aware Security Guardrails**: Built-in input moderation and domain-grounding constraints to block off-topic or explicit queries while supporting objective, clinical scientific responses for medical research papers.
- 📊 **Visual Figure & Table Commentary Mode**: Automatically detects visual query keywords (`figure`, `table`, `graph`, `plot`) and routes to specialized prompt logic for statistical chart and diagram analysis.
- 🔄 **Transparent Query Optimization Breakdown**: Interactive UI expander displaying side-by-side comparison of original user prompts vs. agent-rewritten search queries.
- 📈 **Live Resume Analytics Dashboard**: Real-time tracking of executed queries, indexed PDF chunks, literature reviews generated, and citation precision.
- 👁️ **Multimodal PDF Parsing**: Zero-dependency paragraph text chunking via `PyMuPDF` with native ChromaDB local vector store fallback.
- 🚀 **Dual Deployment Support**: Runs either as a decoupled **FastAPI REST API** or as a 1-click **Streamlit Cloud** standalone application.

---

## 🏗️ System Architecture

```text
                               ┌────────────────────────────────┐
                               │     Streamlit Dashboard /      │
                               │   Streamlit Cloud Interface    │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │ REST API Request / Direct Fallback Invocation │
                       ▼                                               ▼
          ┌─────────────────────────┐                     ┌─────────────────────────┐
          │  FastAPI REST Server    │                     │  Security Guardrail &   │
          │  (/api/v1/query)        │                     │  Input Moderation       │
          └────────────┬────────────┘                     └────────────┬────────────┘
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               ▼
                                  ┌───────────────────────────┐
                                  │   LangGraph Agent Engine  │
                                  │  (Cyclic State Machine)   │
                                  └────────────┬──────────────┘
                                               │
       ┌───────────────────────┬───────────────┴───────────────┬───────────────────────┐
       ▼                       ▼                               ▼                       ▼
┌──────────────┐     ┌───────────────────┐           ┌───────────────────┐   ┌───────────────────┐
│ Retrieve     │ ──▶ │ Document Relevance│ ──(Yes)─▶ │ Generate Answer   │ ▶ │ Hallucination     │
│ Chunks       │     │ Batch Grader      │           │ & Citations       │   │ Evaluator         │
└──────────────┘     └─────────┬─────────┘           └───────────────────┘   └─────────┬─────────┘
                               │ (No)                                                  │ (Not Grounded)
                               ▼                                                       ▼
                     ┌───────────────────┐                                   ┌───────────────────┐
                     │ Transform Query   │ ◀─────────────────────────────────│ Query Refinement  │
                     │ Node (Max 3 Loops)│                                   └───────────────────┘
                     └───────────────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ Resilient Gemini  │
                     │ Model Pool (429)  │
                     └───────────────────┘
```

---

## 🛠️ Tech Stack

- **Agentic Orchestration**: `LangGraph`, `LangChain`
- **LLM Engine**: Google Gemini (`ResilientGeminiLLM` Failover Pool)
- **Vector Database**: `ChromaDB` (with ONNX `all-MiniLM-L6-v2` Fallback)
- **Document Processing**: `PyMuPDF` (`fitz`)
- **Backend REST API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend Dashboard**: `Streamlit`

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get a free key here](https://aistudio.google.com/))

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Aestheticfrog/Multimodal-Agentic-RAG-Research-Assistant.git
cd Multimodal-Agentic-RAG-Research-Assistant

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
BACKEND_URL=http://localhost:8000
```

### 4. Running Locally

**Start the FastAPI Backend Server:**
```bash
uvicorn backend.app.main:app --reload --port 8000
```
*Interactive Swagger OpenAPI docs will be available at `http://localhost:8000/docs`*

**Start the Streamlit Frontend Dashboard:**
```bash
streamlit run frontend/app.py
```
*Access dashboard at `http://localhost:8501`*

---

## ☁️ Deployment

### Streamlit Community Cloud (1-Click Deployment)
1. Fork / Push this repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Set **Main file path** to `frontend/app.py`.
4. In **Settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
   ```
5. Click **Deploy!**

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
