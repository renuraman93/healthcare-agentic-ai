"""
Embedding generation using Gemini's embedding model.

Uses Google's current google-genai SDK.
"""

from google import genai

from app.config.settings import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class GeminiEmbeddings:
    """Wraps Gemini's embedding API."""

    def __init__(self, model_name: str | None = None) -> None:

        if not settings.gemini_api_key:
            raise EmbeddingError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

        self.model_name = (
            model_name or settings.embedding_model_name
        )

        logger.info(
            f"GeminiEmbeddings initialized with model: "
            f"{self.model_name}"
        )

    def embed_text(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> list[float]:
        """
        Embed a single piece of text.

        task_type:
            retrieval_document -> text being stored in vector DB
            retrieval_query    -> user search/query text
        """

        try:

            if not text or not text.strip():
                raise EmbeddingError(
                    "Text cannot be empty."
                )

            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config={
                    "task_type": task_type
                }
            )

            if not result.embeddings:
                raise EmbeddingError(
                    "Gemini returned no embedding."
                )

            return result.embeddings[0].values

        except EmbeddingError:
            raise

        except Exception as exc:

            logger.error(
                f"Embedding failed: {exc}"
            )

            raise EmbeddingError(
                f"Failed to embed text: {exc}"
            ) from exc

    def embed_batch(
        self,
        texts: list[str],
        task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        """
        Embed multiple texts.

        For the current demo-scale implementation we process
        the texts individually.
        """

        if not texts:
            return []

        embeddings = []

        for i, text in enumerate(texts):

            embeddings.append(
                self.embed_text(
                    text,
                    task_type=task_type
                )
            )

            if (i + 1) % 10 == 0:

                logger.info(
                    f"Embedded "
                    f"{i + 1}/{len(texts)} chunks..."
                )

        return embeddings