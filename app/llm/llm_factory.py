"""
Factory for creating the configured LLM provider.

This is the ONLY place in the app that should decide which concrete
LLM class to instantiate. Everything else asks for an LLMProvider
and doesn't care what's behind it.
"""

from app.llm.base import LLMProvider
from app.llm.gemini_llm import GeminiLLM
from app.config.settings import settings


def get_llm() -> LLMProvider:
    """Returns the LLM provider configured in settings."""
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        return GeminiLLM()

    # Future providers plug in here, e.g.:
    # if provider == "openai":
    #     return OpenAILLM()
    # if provider == "local":
    #     return LocalLLM()

    raise ValueError(f"Unsupported LLM provider: '{provider}'")