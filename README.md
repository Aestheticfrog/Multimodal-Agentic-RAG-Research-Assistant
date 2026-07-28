# 🔬 ResearchPilot AI – Agentic Multimodal Research Assistant

**ResearchPilot AI** is an advanced agentic RAG system built to analyze complex research papers, perform literature reviews, and answer questions with grounded citations. Built with **LangGraph**, **Google Gemini**, **ChromaDB**, **FastAPI**, and **Streamlit**.

---

## 🌟 Key Features

- 🧠 **Agentic RAG Engine (LangGraph)**: Integrates **Adaptive RAG**, **Self-RAG**, and **Corrective RAG (CRAG)** to dynamically grade document relevance, rewrite queries, and evaluate hallucinations before delivering answers.
- 👁️ **Multimodal PDF Parsing**: Extracts text, tables, and visual diagram context using `PyMuPDF` for comprehensive paper understanding.
- 📝 **Automated Literature Review Generator**: Generates structured synthesis reports (Executive Summary, Key Findings, Methodology Comparison, Gaps, and Future Directions).
- 🏷️ **Grounding & Citation Breakdown**: Every generated response traces back to specific document sources and page references.
- ⚡ **Asynchronous FastAPI Backend**: Production-ready API endpoints decoupled from the presentation layer.
- 🎨 **Modern Streamlit Dashboard**: Clean, responsive interface featuring live agent workflow visualization and multi-document library management.

---

## 🏗️ Architecture Overview

```text
                     ┌────────────────────────┐
                     │    Streamlit UI        │
                     └───────────┬────────────┘
                                 │ HTTP
                                 ▼
                     ┌────────────────────────┐
                     │    FastAPI Server      │
                     └───────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
    ┌─────────────────────────┐     ┌─────────────────────────┐
    │  LangGraph Agent Engine │     │  ChromaDB Vector Store  │
    │  (Adaptive/Self RAG)    │     └─────────────────────────┘
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Google Gemini 2.5/3.0  │
    └─────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get a free key here](https://aistudio.google.com/))

### 2. Environment Setup

```bash
# Clone repository & navigate to project folder
cd ResearchPilot_AI

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration

Copy `.env.example` to `.env` and fill in your Gemini API Key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 4. Running the Application

**Start the FastAPI Backend:**
```bash
uvicorn backend.app.main:app --reload --port 8000
```
*API Documentation available at `http://localhost:8000/docs`*

**Start the Streamlit Frontend:**
```bash
streamlit run frontend/app.py
```
*Access dashboard at `http://localhost:8501`*

---

## 🧪 Tech Stack

- **Orchestration**: `LangGraph`, `LangChain`
- **LLM & Vision**: `google-genai` / `langchain-google-genai` (`gemini-2.5-flash`)
- **Vector Database**: `ChromaDB`
- **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend UI**: `Streamlit`
- **Document Processing**: `PyMuPDF` (FitZ)
