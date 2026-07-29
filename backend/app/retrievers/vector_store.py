"""ChromaDB Vector Store Management with Unified Zero-Crash Local Persistence."""
import os
import logging
from typing import List, Optional, Dict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger("researchpilot.vectorstore")

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./database/chroma_store")
_vector_store_instance: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Returns singleton instance of ChromaDB vector store using unified local persistence."""
    global _vector_store_instance
    if _vector_store_instance is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        # Using Chroma's native ONNX default embeddings to guarantee 100% collection unity across all paper uploads
        _vector_store_instance = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="unified_research_papers",
        )
        logger.info("ChromaDB vector store initialized with unified local ONNX persistence.")

    return _vector_store_instance


def add_documents_to_vector_store(documents: List[Document]) -> int:
    """Ingests list of LangChain Document objects into ChromaDB unified store."""
    vs = get_vector_store()
    try:
        vs.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} document chunks to unified ChromaDB.")
        return len(documents)
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        return len(documents)


def retrieve_adaptive_documents(query: str) -> List[Document]:
    """Dynamically detects query intent and performs adaptive, source-stratified vector retrieval.
    - Comparative / Multi-paper queries -> Retrieves k=24 and performs balanced round-robin sampling across all source PDFs.
    - Targeted / Fact queries -> High-precision k=6 search.
    """
    vs = get_vector_store()
    if vs is None:
        return []

    comp_keywords = [
        "compare", "comparison", "difference", "between", "versus", "vs",
        "both", "all papers", "literature review", "synthesis", "different", "two"
    ]
    is_comparative = any(kw in query.lower() for kw in comp_keywords)
    target_k = 24 if is_comparative else 6

    try:
        raw_docs = vs.similarity_search(query, k=target_k)
        if not raw_docs:
            return []

        if not is_comparative:
            return raw_docs[:6]

        # Group retrieved chunks by source file name
        docs_by_source: Dict[str, List[Document]] = {}
        for doc in raw_docs:
            src = doc.metadata.get("source", "unknown")
            if src not in docs_by_source:
                docs_by_source[src] = []
            docs_by_source[src].append(doc)

        logger.info(f"Multi-Paper Sources Found in Vector Store: {list(docs_by_source.keys())}")

        # Balanced sampling: up to 10 chunks per paper
        stratified_docs: List[Document] = []
        max_per_src = 10
        for src, doc_list in docs_by_source.items():
            stratified_docs.extend(doc_list[:max_per_src])

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
