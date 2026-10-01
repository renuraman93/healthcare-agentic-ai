"""
Phase 2 sanity check.
Confirms: settings load correctly and logging works.
"""

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting Phase 2 verification...")

    print("\n--- Loaded Settings ---")
    print("App name:       ", settings.app_name)
    print("Environment:    ", settings.app_env)
    print("LLM provider:   ", settings.llm_provider)
    print("LLM model:      ", settings.llm_model_name)
    print("Embedding model:", settings.embedding_model_name)
    print("Vector store:   ", settings.vector_store_dir)
    print("Chunk size/overlap:", settings.chunk_size, "/", settings.chunk_overlap)
    print("API will run on:", f"{settings.api_host}:{settings.api_port}")

    if settings.gemini_api_key:
        logger.info("GEMINI_API_KEY is set (value hidden).")
    else:
        logger.warning("GEMINI_API_KEY is NOT set. LLM calls will fail in Phase 3.")

    logger.info("Phase 2 verification complete.")


if __name__ == "__main__":
    main()