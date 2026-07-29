"""ChromaDB Vector Store Management and Hybrid Adaptive Retriever Initialization."""
import os
import logging
from typing import List, Optional, Dict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from backend.app.models.llm import get_gemini_embeddings

logger = logging.getLogger("researchpilot.vectorstore")

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./database/chroma_db")
_vector_store_instance: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Returns singleton instance of ChromaDB vector store with zero-crash fallback."""
    global _vector_store_instance
    if _vector_store_instance is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        try:
            embeddings = get_gemini_embeddings()
            _vector_store_instance = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="research_papers",
            )
            logger.info("ChromaDB vector store initialized with Gemini embeddings.")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini embeddings: {e}. Falling back to default ONNX embeddings.")
            _vector_store_instance = Chroma(
                persist_directory=f"{PERSIST_DIR}_default",
                collection_name="research_papers_default",
            )
            logger.info("ChromaDB vector store initialized with default ONNX embeddings.")

    return _vector_store_instance


def add_documents_to_vector_store(documents: List[Document]) -> int:
    """Ingests list of LangChain Document objects into ChromaDB."""
    vs = get_vector_store()
    try:
        vs.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} document chunks to ChromaDB.")
        return len(documents)
    except Exception as e:
        logger.warning(f"Embedding error during Chroma ingestion: {e}. Switching to Chroma default ONNX embeddings...")
        fallback_vs = Chroma(
            persist_directory=f"{PERSIST_DIR}_default",
            collection_name="research_papers_default",
        )
        fallback_vs.add_documents(documents)
        global _vector_store_instance
        _vector_store_instance = fallback_vs
        logger.info(f"Successfully added {len(documents)} chunks to default ONNX vector store.")
        return len(documents)


def retrieve_adaptive_documents(query: str) -> List[Document]:
    """Dynamically detects query intent and performs adaptive, source-stratified vector retrieval.
    - Comparative / Multi-paper queries -> Retrieves k=20 and performs balanced sampling across all source PDFs.
    - Targeted / Fact queries -> High-precision k=6 search.
    """
    vs = get_vector_store()
    if vs is None:
        return []

    comp_keywords = [
        "compare", "comparison", "difference", "between", "versus", "vs",
        "both", "all papers", "literature review", "synthesis", "different"
    ]
    is_comparative = any(kw in query.lower() for kw in comp_keywords)
    target_k = 20 if is_comparative else 6

    try:
        raw_docs = vs.similarity_search(query, k=target_k)
        if not raw_docs:
            return []

        if not is_comparative:
            return raw_docs[:6]

        # Stratify by source PDF to ensure multi-paper fairness
        docs_by_source: Dict[str, List[Document]] = {}
        for doc in raw_docs:
            src = doc.metadata.get("source", "unknown")
            if src not in docs_by_source:
                docs_by_source[src] = []
            docs_by_source[src].append(doc)

        stratified_docs: List[Document] = []
        max_per_src = 8
        for src, doc_list in docs_by_source.items():
            stratified_docs.extend(doc_list[:max_per_src])

        logger.info(
            f"Adaptive Retrieval: Comparative={is_comparative}, "
            f"Sources Found={list(docs_by_source.keys())}, "
            f"Total Chunks={len(stratified_docs)}"
        )
        return stratified_docs if stratified_docs else raw_docs
    except Exception as e:
        logger.error(f"Error in adaptive document retrieval: {e}")
        return []


def get_retriever(k: int = 16) -> Optional[VectorStoreRetriever]:
    """Returns LangChain retriever interface for vector store."""
    try:
        vs = get_vector_store()
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Error getting retriever: {e}")
        return None
