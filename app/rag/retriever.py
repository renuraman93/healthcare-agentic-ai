"""
Retriever: takes a user query, embeds it, and returns the top-K
most relevant chunks with their source metadata.
"""

from pydantic import BaseModel

from app.rag.vector_store import VectorStore
from app.rag.embeddings import GeminiEmbeddings
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievedChunk(BaseModel):
    text: str
    source: str
    document_type: str
    chunk_index: int
    relevance_distance: float


class Retriever:
    def __init__(self) -> None:
        self._store = VectorStore()
        self._embedder = GeminiEmbeddings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k

        if self._store.count() == 0:
            logger.warning("Vector store is empty — no documents indexed yet.")
            return []

        query_embedding = self._embedder.embed_text(query, task_type="retrieval_query")
        results = self._store.query(query_embedding=query_embedding, top_k=top_k)

        chunks: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    document_type=meta.get("document_type", "unknown"),
                    chunk_index=meta.get("chunk_index", -1),
                    relevance_distance=dist,
                )
            )

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Builds a single context string from retrieved chunks, with
        source labels, for injection into an LLM prompt.
        """
        if not chunks:
            return ""

        parts = []
        for c in chunks:
            parts.append(f"[Source: {c.source}, chunk {c.chunk_index}]\n{c.text}")
        return "\n\n---\n\n".join(parts)