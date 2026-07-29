"""LangGraph Agentic Workflow Compilation for ResearchPilot AI."""
import logging
from typing import List, Optional

from langgraph.graph import END, StateGraph
from langchain_core.documents import Document

from backend.app.agents.state import AgentState
from backend.app.agents.nodes import (
    grade_documents_node,
    generate_answer_node,
    transform_query_node,
    web_search_node,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)

logger = logging.getLogger("researchpilot.graph")


def retrieve_docs_node(state: AgentState) -> AgentState:
    """Retrieves documents from Vector Store and merges with in-memory session docs."""
    logger.info("--- NODE: RETRIEVE DOCUMENTS ---")
    question = state["question"]
    existing_docs = state.get("documents", [])

    from backend.app.retrievers.vector_store import retrieve_adaptive_documents
    vector_docs = retrieve_adaptive_documents(question)

    combined = list(existing_docs) + list(vector_docs)
    seen = set()
    dedup_docs = []
    for d in combined:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:50])
        if key not in seen:
            seen.add(key)
            dedup_docs.append(d)

    return {
        **state,
        "documents": dedup_docs,
        "retry_count": state.get("retry_count", 0),
    }


def build_researchpilot_graph() -> StateGraph:
    """Builds and compiles the Adaptive / Self-RAG state graph."""
    workflow = StateGraph(AgentState)

    # 1. Define Nodes
    workflow.add_node("retrieve", retrieve_docs_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search", web_search_node)

    # 2. Build Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional Routing: After document grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate_answer": "generate_answer",
        },
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate_answer")

    # Conditional Routing: After answer generation (Hallucination check)
    workflow.add_conditional_edges(
        "generate_answer",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not_useful": "transform_query",
            "max_retries": END,
        },
    )

    return workflow.compile()


# Compiled Singleton Graph Instance
researchpilot_agent = build_researchpilot_graph()
