"""
POST /documents/upload, POST /documents/index, GET /documents
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.api import (
    DocumentInfo,
    DocumentListResponse,
    IndexRequest,
    IndexResponse,
    UploadResponse,
)
from app.services.document_service import DocumentServiceError
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    service = request.app.state.document_service
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        saved_path = service.save_upload(file.filename, content)
    except DocumentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return UploadResponse(
        filename=saved_path.name,
        saved_path=str(saved_path),
        size_bytes=len(content),
    )


@router.post("/documents/index", response_model=IndexResponse)
def index_document(payload: IndexRequest, request: Request) -> IndexResponse:
    service = request.app.state.document_service

    try:
        count = service.index_document(payload.filename)
    except DocumentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"/documents/index unexpected failure: {type(exc).__name__}")
        raise HTTPException(status_code=500, detail="Failed to index document.")

    return IndexResponse(filename=payload.filename, chunks_indexed=count)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    service = request.app.state.document_service
    sources = service.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(source=s) for s in sources],
        total_chunks=service.total_chunks(),
    )