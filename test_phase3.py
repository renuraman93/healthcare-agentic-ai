"""
Phase 3 sanity check.
Confirms: LLM factory + Gemini provider can generate a real response.
"""

from app.llm.llm_factory import get_llm
from app.llm.base import LLMGenerationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Creating LLM provider...")

    try:
        llm = get_llm()
    except LLMGenerationError as e:
        logger.error(f"Could not initialize LLM: {e}")
        return

    system_prompt = (
        "You are a careful healthcare information assistant. "
        "You are not a doctor and do not diagnose patients."
    )
    prompt = "In one sentence, explain what hypertension is."

    logger.info("Sending test prompt to Gemini...")
    try:
        response = llm.generate(prompt=prompt, system_prompt=system_prompt)
        print("\n--- Gemini Response ---")
        print(response)
    except LLMGenerationError as e:
        logger.error(f"Generation failed: {e}")


if __name__ == "__main__":
    main()