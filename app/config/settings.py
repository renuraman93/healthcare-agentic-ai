"""
Centralized application configuration.

Every environment variable the app needs is declared here, with a type
and (where sensible) a default. Import `settings` anywhere in the app
instead of calling os.getenv() directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM Provider ---
    gemini_api_key: str = ""
    llm_provider: str = "gemini"          # "gemini" | future: "openai", "local"
    llm_model_name: str = "gemini-3.5-flash-lite"

    # --- Embeddings ---
    embedding_provider: str = "gemini"    # "gemini" | "local"
    embedding_model_name: str = "gemini-embedding-001"

    # --- Vector Store ---
    vector_store_dir: str = "./vectorstore"
    vector_collection_name: str = "healthcare_docs"

    # --- RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    relevance_max_distance: float = 0.35   # chunks with distance above this are treated as "not relevant"
    use_llm_router: bool = False   # False = rule-based routing only (no LLM calls for routing)
    # --- App / API ---
    app_name: str = "Healthcare Agentic AI Chatbot"
    app_env: str = "development"          # "development" | "production"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- File Upload Limits ---
    max_upload_size_mb: int = 10
    allowed_upload_extensions: str = ".pdf,.txt,.docx"

    # --- Logging ---
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance so .env is parsed only once
    per process, not on every import.
    """
    return Settings()


settings = get_settings()