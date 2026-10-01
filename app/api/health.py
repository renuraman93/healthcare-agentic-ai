"""
GET /health — basic liveness and configuration check.
"""

from fastapi import APIRouter, Request

from app.config.settings import settings
from app.models.api import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    doc_service = request.app.state.document_service
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        llm_provider=settings.llm_provider,
        vector_store_chunks=doc_service.total_chunks(),
    )