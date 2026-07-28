"""Graph nodes and conditional edge logic for LangGraph agent workflow."""
import logging
from typing import List, Literal, Any
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from backend.app.agents.state import AgentState
from backend.app.agents.prompts import (
    DOCUMENT_GRADER_PROMPT,
    ANSWER_GENERATOR_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    ANSWER_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    FIGURE_ANALYZER_PROMPT,
)
from backend.app.models.llm import get_gemini_llm

logger = logging.getLogger("researchpilot.nodes")


def extract_text_content(content: Any) -> str:
    """Safely extracts text content from string or list of content blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)
    return str(content)


# Pydantic Schemas for Structured Output Grading
class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if document is relevant, 'no' if not."
    )


class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(
        description="Grounding score: 'yes' if generation is grounded in facts, 'no' if not."
    )


class GradeAnswer(BaseModel):
    """Binary score for answer quality check."""
    binary_score: str = Field(
        description="Usefulness score: 'yes' if answer resolves user question, 'no' if not."
    )


def grade_documents_node(state: AgentState) -> AgentState:
    """Evaluates whether retrieved documents are relevant to the question in a SINGLE batch LLM call.
    Prevents API rate limits (429) while ensuring strict relevance filtering.
    """
    logger.info("--- NODE: GRADE DOCUMENTS (BATCH MODE) ---")
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.info("No documents retrieved. Web search needed.")
        return {**state, "web_search_needed": True}

    try:
        llm = get_gemini_llm(model_name="gemini-flash-latest", temperature=0.0)
        formatted_docs = "\n---\n".join([f"Document {i+1}:\n{doc.page_content[:400]}" for i, doc in enumerate(documents)])

        prompt = f"""You are a research evaluator. Determine if the retrieved documents contain information relevant to answer the user question.

User Question: {question}

Retrieved Documents:
{formatted_docs}

Evaluate overall relevance:
If at least one document mentions concepts related to the user question, respond 'yes'.
If NONE of the documents are relevant at all, respond 'no'.
Respond ONLY with 'yes' or 'no'."""

        res = llm.invoke(prompt)
        content = extract_text_content(res.content).strip().lower() if hasattr(res, "content") else str(res).strip().lower()

        if "no" in content and "yes" not in content:
            logger.info("Batch document grading marked documents as irrelevant. Web search needed.")
            return {**state, "web_search_needed": True}
        else:
            logger.info("Batch document grading confirmed relevant context.")
            return {**state, "web_search_needed": False}

    except Exception as e:
        logger.warning(f"Error in batch document grading: {e}. Keeping documents by default.")
        return {**state, "web_search_needed": False}


def generate_answer_node(state: AgentState) -> AgentState:
    """Synthesizes answer from relevant documents using Gemini LLM.
    Automatically switches to FIGURE_ANALYZER_PROMPT if the user asks specifically about figures, graphs, or charts.
    """
    logger.info("--- NODE: GENERATE ANSWER ---")
    question = state["question"]
    documents = state.get("documents", [])

    llm = get_gemini_llm(temperature=0.2)

    # Format context with citations metadata
    formatted_context_list = []
    citations = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", f"Document {idx}")
        page = doc.metadata.get("page", "N/A")
        has_img = " (Contains Visual Figures/Tables)" if doc.metadata.get("has_images") else ""
        formatted_context_list.append(f"[{idx}] Source: {source} (Page {page}){has_img}\nContent: {doc.page_content}\n")
        citations.append({"id": idx, "source": source, "page": page})

    formatted_context = "\n---\n".join(formatted_context_list)

    # Detect if user query is specifically targeting figures/graphs/charts
    figure_keywords = ["figure", "fig", "graph", "chart", "diagram", "plot", "table", "visual", "illustration"]
    is_figure_focused = any(kw in question.lower() for kw in figure_keywords)

    if is_figure_focused:
        logger.info("User query targets figures/graphs. Using FIGURE_ANALYZER_PROMPT.")
        prompt = ChatPromptTemplate.from_template(FIGURE_ANALYZER_PROMPT)
    else:
        prompt = ChatPromptTemplate.from_template(ANSWER_GENERATOR_PROMPT)

    formatted_prompt = prompt.format(question=question, context=formatted_context)
    response = llm.invoke(formatted_prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)
    generation_text = extract_text_content(raw_content)

    return {
        **state,
        "generation": generation_text,
        "citations": citations,
    }


def transform_query_node(state: AgentState) -> AgentState:
    """Rewrites user query to optimize vector DB search recall."""
    logger.info("--- NODE: TRANSFORM QUERY ---")
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    llm = get_gemini_llm(temperature=0.4)
    prompt = ChatPromptTemplate.from_template(QUERY_REWRITER_PROMPT)
    formatted_prompt = prompt.format(question=question)

    response = llm.invoke(formatted_prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)
    better_query = extract_text_content(raw_content)

    logger.info(f"Query rewritten: '{question}' -> '{better_query}'")

    return {
        **state,
        "question": better_query.strip(),
        "retry_count": retry_count + 1,
    }


def web_search_node(state: AgentState) -> AgentState:
    """Fallback web search node for out-of-knowledge domain queries."""
    logger.info("--- NODE: WEB SEARCH ---")
    question = state["question"]
    documents = state.get("documents", [])

    # Placeholder for web search integration (Tavily/DuckDuckGo)
    search_doc = Document(
        page_content=f"Web search results for research query: '{question}'. Additional evidence retrieved from online academic databases.",
        metadata={"source": "Web Search Engine", "page": "Online"}
    )
    documents.append(search_doc)

    return {
        **state,
        "documents": documents,
        "web_search_needed": False,
    }


# Conditional Edge Router Functions
def decide_to_generate(state: AgentState) -> Literal["transform_query", "generate_answer"]:
    """Determines whether to generate answer or rewrite query based on doc relevance."""
    web_search_needed = state.get("web_search_needed", False)
    retry_count = state.get("retry_count", 0)

    if web_search_needed and retry_count < 2:
        logger.info("Routing to transform_query node.")
        return "transform_query"
    else:
        logger.info("Routing to generate_answer node.")
        return "generate_answer"


def grade_generation_v_documents_and_question(
    state: AgentState
) -> Literal["useful", "not_useful", "max_retries"]:
    """Evaluates hallucination & answer quality to decide whether to output or retry."""
    logger.info("--- EVALUATING GENERATION GROUNDING & QUALITY ---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)

    if retry_count >= 2:
        logger.warning("Max retries reached. Returning generation.")
        return "useful"

    llm = get_gemini_llm(temperature=0.0)

    # Check hallucination grounding
    try:
        structured_hallucination = llm.with_structured_output(GradeHallucination)
        prompt_h = ChatPromptTemplate.from_template(HALLUCINATION_GRADER_PROMPT)
        chain_h = prompt_h | structured_hallucination
        formatted_context = "\n\n".join([d.page_content for d in documents])
        res_h: GradeHallucination = chain_h.invoke({"context": formatted_context, "generation": generation})

        if res_h.binary_score.lower() == "yes":
            logger.info("Generation is grounded in documents.")
            # Check answer usefulness
            structured_answer = llm.with_structured_output(GradeAnswer)
            prompt_a = ChatPromptTemplate.from_template(ANSWER_GRADER_PROMPT)
            chain_a = prompt_a | structured_answer
            res_a: GradeAnswer = chain_a.invoke({"question": question, "generation": generation})

            if res_a.binary_score.lower() == "yes":
                logger.info("Answer is useful and addresses question.")
                return "useful"
            else:
                logger.info("Answer is not useful. Rewriting query.")
                return "not_useful"
        else:
            logger.info("Generation contains ungrounded statements. Retrying.")
            return "not_useful"
    except Exception as e:
        logger.warning(f"Error during generation grading: {e}. Defaulting to useful.")
        return "useful"
