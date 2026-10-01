"""
Shared state passed between LangGraph nodes.

Each node returns a dict of ONLY the keys it wants to update;
LangGraph merges that into the running state.
"""

from typing import TypedDict

from app.models.schemas import AgentResult, SourceRef


class GraphState(TypedDict, total=False):
    # Input
    request_id: str
    question: str
    history: list[dict]          # [{"role": "user"|"assistant", "content": "..."}]

    # Supervisor output
    route: str                   # document | clinical_extraction | summarization | general
    emergency: bool
    retrieval_query: str         # question, enriched with prior turn for follow-ups

    # Document search output
    doc_found: bool
    context: str
    sources: list[SourceRef]

    # Agent output
    answer: str
    agent: str
    extraction: dict

    # Error handling
    error: str                   # exception type name, never message text
    failed_node: str

    # Final output
    result: AgentResult