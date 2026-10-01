"""
Document service: orchestrates the ingestion side of the RAG pipeline
(load -> clean -> chunk -> store) for use by the API layer.
"""

from pathlib import Path
import uuid

from app.config.settings import settings
from app.rag.loaders import load_document, DocumentLoadError
from app.rag.cleaning import clean_text
from app.rag.chunking import chunk_text
from app.rag.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = Path("data/documents")


class DocumentServiceError(Exception):
    """Raised for any ingestion failure the API should surface cleanly."""
    pass


class DocumentService:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store or VectorStore()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> Path:
        """Validates and saves raw uploaded bytes to disk. Returns the saved path."""
        safe_name = Path(filename).name  # strips any directory traversal attempt
        ext = Path(safe_name).suffix.lower()

        allowed = [e.strip().lower() for e in settings.allowed_upload_extensions.split(",")]
        if ext not in allowed:
            raise DocumentServiceError(
                f"Unsupported file type '{ext}'. Allowed: {allowed}"
            )

        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise DocumentServiceError(
                f"File too large: {size_mb:.2f}MB (max {settings.max_upload_size_mb}MB)"
            )

        dest = UPLOAD_DIR / safe_name
        if dest.exists():
            stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
            dest = UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        dest.write_bytes(content)
        logger.info(f"Saved upload: {safe_name} ({len(content)} bytes)")
        return dest

    def index_document(self, filename: str) -> int:
        """Loads, cleans, chunks, and stores a previously-saved document. Returns chunk count."""
        safe_name = Path(filename).name
        file_path = UPLOAD_DIR / safe_name

        if not file_path.exists():
            raise DocumentServiceError(f"File not found in uploads: {safe_name}")

        try:
            document = load_document(str(file_path))
        except DocumentLoadError as exc:
            raise DocumentServiceError(str(exc)) from exc

        cleaned = clean_text(document.full_text)
        chunks = chunk_text(cleaned)

        if not chunks:
            raise DocumentServiceError(f"No content to index in: {safe_name}")

        chunk_texts = [c.text for c in chunks]
        count = self._store.add_chunks(
            chunks=chunk_texts,
            source_filename=document.source_filename,
            document_type=document.document_type,
        )
        return count

    def list_documents(self) -> list[str]:
        return self._store.list_sources()

    def total_chunks(self) -> int:
        return self._store.count()