"""
ChromaDB vector store: stores chunks + embeddings + metadata,
and provides similarity search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import settings
from app.rag.embeddings import GeminiEmbeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.vector_store_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.vector_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = GeminiEmbeddings()
        logger.info(
            f"VectorStore ready. Collection: '{settings.vector_collection_name}' "
            f"at '{settings.vector_store_dir}'"
        )

    def add_chunks(
        self,
        chunks: list[str],
        source_filename: str,
        document_type: str,
        extra_metadata: dict | None = None,
    ) -> int:
        """
        Embeds and stores a list of text chunks for one document.
        Returns the number of chunks added.
        """
        if not chunks:
            logger.warning(f"No chunks to add for {source_filename}")
            return 0

        embeddings = self._embedder.embed_batch(chunks, task_type="retrieval_document")

        ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        for i in range(len(chunks)):
            meta = {
                "source": source_filename,
                "document_type": document_type,
                "chunk_index": i,
            }
            if extra_metadata:
                meta.update(extra_metadata)
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks from '{source_filename}' to vector store")
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Returns the unique set of source filenames currently indexed."""
        results = self._collection.get(include=["metadatas"])
        sources = {m["source"] for m in results["metadatas"] if "source" in m}
        return sorted(sources)

    def delete_source(self, source_filename: str) -> None:
        """Removes all chunks belonging to a given source document."""
        self._collection.delete(where={"source": source_filename})
        logger.info(f"Deleted all chunks for source: {source_filename}")

    def query(
        self, query_embedding: list[float], top_k: int
    ) -> dict:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )