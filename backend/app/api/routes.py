"""FastAPI API Routes for document uploading, research querying, and literature review generation."""
import logging
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.app.agents.graph import researchpilot_agent
from backend.app.retrievers.vector_store import add_documents_to_vector_store, get_retriever
from backend.app.utils.pdf_parser import parse_pdf_bytes
from backend.app.models.llm import get_gemini_llm

logger = logging.getLogger("researchpilot.api")

router = APIRouter(prefix="/api/v1", tags=["ResearchPilot Engine"])


# Request & Response Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the key mechanisms of Adaptive RAG?")


class CitationItem(BaseModel):
    id: int
    source: str
    page: str


class QueryResponse(BaseModel):
    question: str
    original_question: Optional[str] = None
    generation: str
    citations: List[CitationItem]
    retry_count: int


class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks_indexed: int
    message: str


class LiteratureReviewRequest(BaseModel):
    topic: str = Field(..., example="Agentic RAG and Self-Correction Mechanisms in LLMs")


class LiteratureReviewResponse(BaseModel):
    topic: str
    executive_summary: str
    full_report: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Endpoint to upload a PDF research paper, parse it, and index its vectors into ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        content = await file.read()
        documents = parse_pdf_bytes(content, filename=file.filename)
        if not documents:
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        indexed_count = add_documents_to_vector_store(documents)
        return UploadResponse(
            filename=file.filename,
            status="success",
            chunks_indexed=indexed_count,
            message=f"Successfully indexed '{file.filename}' ({indexed_count} chunks)."
        )
    except Exception as e:
        logger.error(f"Error processing PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Executes the Agentic RAG state graph (LangGraph) for a user question."""
    try:
        initial_state = {
            "question": request.question,
            "original_question": request.question,
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "hallucination_grade": "",
            "answer_grade": "",
            "retry_count": 0,
            "citations": [],
        }

        final_state = researchpilot_agent.invoke(initial_state)

        citations_data = [
            CitationItem(
                id=c.get("id", idx),
                source=str(c.get("source", "Unknown")),
                page=str(c.get("page", "N/A"))
            )
            for idx, c in enumerate(final_state.get("citations", []), start=1)
        ]

        return QueryResponse(
            question=final_state.get("question", request.question),
            original_question=request.question,
            generation=final_state.get("generation", "No generation produced."),
            citations=citations_data,
            retry_count=final_state.get("retry_count", 0),
        )
    except Exception as e:
        logger.error(f"Error executing agent workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@router.post("/literature-review", response_model=LiteratureReviewResponse)
async def generate_literature_review(request: LiteratureReviewRequest):
    """Generates an automated, structured literature review synthesis from ingested vector documents."""
    try:
        retriever = get_retriever(k=8)
        context_docs = retriever.invoke(request.topic) if retriever else []

        formatted_context = "\n---\n".join([d.page_content for d in context_docs]) if context_docs else "No specific documents indexed yet."

        llm = get_gemini_llm(temperature=0.3)
        prompt = f"""You are a senior research professor writing a Literature Review Synthesis Report on the topic: '{request.topic}'.

Available Research Evidence:
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
        from backend.app.agents.nodes import extract_text_content
        report_text = extract_text_content(raw_report)

        # Extract first 2 paragraphs as executive summary
        paragraphs = report_text.strip().split("\n\n")
        exec_summary = "\n\n".join(paragraphs[:2]) if len(paragraphs) >= 2 else report_text[:300]

        return LiteratureReviewResponse(
            topic=request.topic,
            executive_summary=exec_summary,
            full_report=report_text,
        )
    except Exception as e:
        logger.error(f"Error generating literature review: {e}")
        raise HTTPException(status_code=500, detail=f"Literature review generation failed: {str(e)}")
