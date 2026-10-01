"""
Shared Pydantic schemas used across the ingestion, RAG, and agent layers.
"""

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """
    Represents one fully-loaded source document, before chunking.
    """
    source_filename: str = Field(..., description="Original filename, e.g. 'report.pdf'")
    document_type: str = Field(..., description="File extension without dot, e.g. 'pdf'")
    full_text: str = Field(..., description="All extracted text from the document")
    page_count: int | None = Field(default=None, description="Number of pages, if applicable")
    char_count: int = Field(default=0, description="Length of full_text")


class SourceRef(BaseModel):
    """A reference to where a piece of information came from."""
    source: str
    chunk_index: int
    distance: float | None = None


class AgentResult(BaseModel):
    """The final, user-facing result returned by the pipeline."""
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    agent: str
    emergency_flag: bool = False