"""
Thin HTTP client for the FastAPI backend. Streamlit calls ONLY this
module — never the app/ package directly — so the frontend has no
business logic of its own.
"""

import httpx

DEFAULT_TIMEOUT = 30.0


class ApiError(Exception):
    """Raised when the backend returns an error or is unreachable."""
    pass


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict:
        try:
            resp = httpx.get(f"{self.base_url}{path}", timeout=DEFAULT_TIMEOUT)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Could not reach the API at {self.base_url}. "
                "Is the FastAPI server running?"
            ) from exc
        return self._handle_response(resp)

    def _post(self, path: str, json: dict | None = None, files: dict | None = None) -> dict:
        try:
            resp = httpx.post(
                f"{self.base_url}{path}", json=json, files=files, timeout=DEFAULT_TIMEOUT
            )
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Could not reach the API at {self.base_url}. "
                "Is the FastAPI server running?"
            ) from exc
        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: httpx.Response) -> dict:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(detail)
        return resp.json()

    # ---- endpoint wrappers ----

    def health(self) -> dict:
        return self._get("/health")

    def list_documents(self) -> dict:
        return self._get("/documents")

    def upload_document(self, filename: str, file_bytes: bytes) -> dict:
        files = {"file": (filename, file_bytes)}
        return self._post("/documents/upload", files=files)

    def index_document(self, filename: str) -> dict:
        return self._post("/documents/index", json={"filename": filename})

    def chat(self, message: str, session_id: str) -> dict:
        return self._post("/chat", json={"message": message, "session_id": session_id})