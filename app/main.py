"""
FastAPI application entry point.

Shared, expensive-to-create objects (vector store, document service,
LangGraph workflow) are built once at startup and stored on app.state,
so every request reuses them instead of re-initializing per call.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, health
from app.config.settings import settings
from app.graph.workflow import HealthcareWorkflow
from app.rag.vector_store import VectorStore
from app.services.document_service import DocumentService
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}...")

    vector_store = VectorStore()
    app.state.document_service = DocumentService(vector_store=vector_store)
    app.state.workflow = HealthcareWorkflow()

    logger.info("Startup complete: vector store, document service, and workflow ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    description="Agentic AI healthcare chatbot (educational demo, not medical advice).",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # Turns FastAPI's default (verbose, implementation-revealing) 422 body
    # into a clean, consistent shape matching our other error responses.
    logger.warning(f"Validation error on {request.url.path}: {len(exc.errors())} field(s)")
    return JSONResponse(status_code=422, content={"detail": "Invalid request data."})


@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running. See /docs for the API."}


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)