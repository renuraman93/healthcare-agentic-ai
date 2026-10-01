"""
Document Search Agent: retrieves relevant chunks from the vector store
and decides whether anything relevant was actually found.
"""

from pydantic import BaseModel, Field

from app.config.settings import settings
from app.models.schemas import SourceRef
from app.rag.retriever import Retriever, RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSearchResult(BaseModel):
    found: bool
    context: str = ""
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


class DocumentAgent:
    name = "document_agent"

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or Retriever()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        apply_threshold: bool = True,
    ) -> DocumentSearchResult:
        chunks = self._retriever.retrieve(query, top_k=top_k)

        if apply_threshold:
            relevant = [
                c for c in chunks
                if c.relevance_distance <= settings.relevance_max_distance
            ]
        else:
            # Used for summarization: "summarize the document" is not
            # semantically close to any single chunk, so don't filter.
            relevant = chunks

        if not relevant:
            logger.info(
                f"{self.name}: no relevant chunks "
                f"({len(chunks)} retrieved, threshold applied={apply_threshold})"
            )
            return DocumentSearchResult(found=False)

        sources = [
            SourceRef(
                source=c.source,
                chunk_index=c.chunk_index,
                distance=round(c.relevance_distance, 4),
            )
            for c in relevant
        ]
        context = self._retriever.build_context(relevant)

        logger.info(f"{self.name}: {len(relevant)} relevant chunks found")
        return DocumentSearchResult(
            found=True, context=context, chunks=relevant, sources=sources
        )