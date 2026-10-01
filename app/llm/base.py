"""
Abstract LLM provider interface.

Every concrete LLM implementation (Gemini, OpenAI, local models, etc.)
must implement this interface. Agents and services depend ONLY on this
abstraction, never on a specific provider's SDK.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class all LLM providers must implement."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a text response for a given prompt.

        Args:
            prompt: The user-facing input/question.
            system_prompt: Optional instruction that sets the model's
                behavior/persona (e.g., "You are a careful clinical assistant").

        Returns:
            The generated text response as a string.

        Raises:
            LLMGenerationError: If the underlying API call fails.
        """
        raise NotImplementedError


class LLMGenerationError(Exception):
    """Raised when an LLM provider fails to generate a response."""
    pass