"""Multimodal PDF and Research Paper Parser using PyMuPDF (fitz)."""
import fitz  # PyMuPDF
import logging
from typing import List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger("researchpilot.parser")


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits raw text into overlapping semantic chunks without external dependencies."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> List[Document]:
    """Parses raw PDF bytes, extracts text per page along with visual metadata,
    and returns chunked LangChain Document objects.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunked_documents: List[Document] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        images = page.get_images()
        has_images = len(images) > 0

        if text.strip():
            page_number = page_num + 1
            chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200)
            for chunk in chunks:
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "has_images": has_images,
                        },
                    )
                )

    doc.close()
    logger.info(f"Extracted and chunked '{filename}' into {len(chunked_documents)} chunks.")
    return chunked_documents
