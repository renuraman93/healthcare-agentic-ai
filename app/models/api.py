"""
Pydantic request/response models for the FastAPI layer.
Kept separate from app/models/schemas.py, which holds internal
pipeline models (LoadedDocument, AgentResult, SourceRef).
"""

from pydantic import BaseModel, Field

from app.models.schemas import SourceRef


# ---- /chat ----

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=100)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    agent: str
    emergency_flag: bool


# ---- /documents/upload ----

class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    size_bytes: int


# ---- /documents/index ----

class IndexRequest(BaseModel):
    filename: str = Field(..., min_length=1)


class IndexResponse(BaseModel):
    filename: str
    chunks_indexed: int


# ---- /documents ----

class DocumentInfo(BaseModel):
    source: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


# ---- /health ----

class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm_provider: str
    vector_store_chunks: int


# ---- shared error shape ----

class ErrorResponse(BaseModel):
    detail: str