"""Agent state definition for LangGraph workflow."""
from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    """Represents the shared state of the Agentic RAG graph."""
    
    question: str
    original_question: Optional[str]
    documents: List[Document]
    generation: str
    web_search_needed: bool
    hallucination_grade: str  # 'grounded' or 'not_grounded'
    answer_grade: str         # 'useful' or 'not_useful'
    retry_count: int
    citations: List[dict]
