"""
Shared pytest fixtures. Available automatically to every test file
in this folder — no import needed.
"""

import shutil
from pathlib import Path

import pytest

from app.rag.vector_store import VectorStore
from tests.fakes import FakeLLM


class FakeEmbedder:
    """Deterministic fake embeddings: same text -> same vector, no API calls."""

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        # A cheap deterministic "embedding": length-based, just needs to be
        # consistent and comparable for ChromaDB, not semantically meaningful.
        seed = sum(ord(c) for c in text) % 1000
        return [(seed + i) / 1000 for i in range(16)]

    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        return [self.embed_text(t, task_type) for t in texts]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def failing_llm() -> FakeLLM:
    return FakeLLM(should_fail=True)


@pytest.fixture
def test_vector_store(tmp_path, monkeypatch):
    """
    A VectorStore backed by a throwaway directory and FakeEmbedder,
    so tests never touch your real ./vectorstore or the embedding API.
    """
    test_dir = tmp_path / "test_vectorstore"
    monkeypatch.setattr("app.rag.vector_store.settings.vector_store_dir", str(test_dir))
    monkeypatch.setattr("app.rag.vector_store.settings.vector_collection_name", "test_collection")

    store = VectorStore.__new__(VectorStore)  # bypass __init__'s real GeminiEmbeddings
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    store._client = chromadb.PersistentClient(
        path=str(test_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    store._collection = store._client.get_or_create_collection(
        name="test_collection", metadata={"hnsw:space": "cosine"}
    )
    store._embedder = FakeEmbedder()

    yield store

    shutil.rmtree(test_dir, ignore_errors=True)