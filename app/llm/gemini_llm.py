import time
import logging

from google import genai

from app.config.settings import settings
from app.llm.base import LLMProvider, LLMGenerationError


logger = logging.getLogger(__name__)


class GeminiLLM(LLMProvider):

    def __init__(self, model_name: str | None = None) -> None:

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.model_name = (
            model_name or settings.llm_model_name
        )

        self._client = genai.Client(
            api_key=settings.gemini_api_key
        )

        logger.info(
            f"GeminiLLM initialized with model: "
            f"{self.model_name}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_retries: int = 4,
    ) -> str:

        if system_prompt:
            full_prompt = (
                f"System instructions:\n"
                f"{system_prompt}\n\n"
                f"User request:\n"
                f"{prompt}"
            )
        else:
            full_prompt = prompt

        for attempt in range(max_retries):

            try:

                logger.info(
                    f"Gemini generation attempt "
                    f"{attempt + 1}/{max_retries}"
                )

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )

                if not response or not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response"
                    )

                return response.text

            except Exception as e:

                error_text = str(e)

                # ---------------------------------------------
                # QUOTA EXCEEDED
                # ---------------------------------------------
                if (
                    "429" in error_text
                    and (
                        "RESOURCE_EXHAUSTED"
                        in error_text
                        or "quota"
                        in error_text.lower()
                        or "exceeded"
                        in error_text.lower()
                    )
                ):

                    logger.error(
                        "Gemini API quota exceeded. "
                        "Stopping retries."
                    )

                    raise LLMGenerationError(
                        "Gemini API quota exceeded. "
                        "Please wait for the quota window "
                        "to reset or check your Gemini API "
                        "plan and billing."
                    ) from e

                # ---------------------------------------------
                # TEMPORARY SERVER ERROR
                # ---------------------------------------------
                temporary_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "500" in error_text
                    or "INTERNAL" in error_text
                )

                if (
                    temporary_error
                    and attempt < max_retries - 1
                ):

                    delay = 2 ** attempt

                    logger.warning(
                        f"Gemini temporarily unavailable "
                        f"(attempt {attempt + 1}/"
                        f"{max_retries}). "
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(delay)
                    continue

                logger.error(
                    f"Gemini generation failed: {e}"
                )

                raise LLMGenerationError(
                    f"Gemini API call failed: {e}"
                ) from e

