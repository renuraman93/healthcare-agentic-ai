"""
API tests using FastAPI's TestClient. The real workflow and document
service are replaced with fakes via dependency overrides at app.state,
so these tests make zero Gemini/embedding API calls.
"""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import AgentResult, SourceRef


class FakeWorkflow:
    def run(self, question: str, session_id: str = "default") -> AgentResult:
        if "fail" in question.lower():
            raise RuntimeError("simulated failure")
        return AgentResult(
            answer="Fake answer for: " + question,
            sources=[SourceRef(source="doc.txt", chunk_index=0, distance=0.1)],
            agent="fake_agent",
            emergency_flag="chest pain" in question.lower(),
        )


class FakeDocumentService:
    def __init__(self):
        self.saved = {}

    def save_upload(self, filename, content):
        from pathlib import Path
        self.saved[filename] = content
        return Path(f"data/documents/{filename}")

    def index_document(self, filename):
        if filename not in self.saved:
            from app.services.document_service import DocumentServiceError
            raise DocumentServiceError(f"File not found in uploads: {filename}")
        return 3

    def list_documents(self):
        return list(self.saved.keys())

    def total_chunks(self):
        return len(self.saved) * 3


@pytest.fixture
def client():
    with TestClient(app) as c:
        app.state.workflow = FakeWorkflow()
        app.state.document_service = FakeDocumentService()
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_returns_answer(client):
    resp = client.post("/chat", json={"message": "What is hypertension?", "session_id": "t1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Fake answer" in data["answer"]
    assert data["agent"] == "fake_agent"
    assert data["sources"][0]["source"] == "doc.txt"


def test_chat_empty_message_returns_422(client):
    resp = client.post("/chat", json={"message": "", "session_id": "t1"})
    assert resp.status_code == 422


def test_chat_missing_body_returns_422(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_unexpected_failure_returns_500_not_traceback(client):
    resp = client.post("/chat", json={"message": "please fail", "session_id": "t1"})
    assert resp.status_code == 500
    assert "detail" in resp.json()


def test_upload_and_index_roundtrip(client):
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"some content"), "text/plain")},
    )
    assert upload_resp.status_code == 200
    filename = upload_resp.json()["filename"]

    index_resp = client.post("/documents/index", json={"filename": filename})
    assert index_resp.status_code == 200
    assert index_resp.json()["chunks_indexed"] == 3

    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    assert filename in [d["source"] for d in list_resp.json()["documents"]]


def test_index_nonexistent_file_returns_400(client):
    resp = client.post("/documents/index", json={"filename": "nope.txt"})
    assert resp.status_code == 400


def test_upload_empty_file_returns_400(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 400