"""
A fake LLM that returns canned responses, so tests run without any API
calls or quota. Can also be told to raise, to test error-handling paths.
"""

from app.llm.base import LLMProvider, LLMGenerationError


class FakeLLM(LLMProvider):
    def __init__(self, should_fail: bool = False) -> None:
        self.calls = 0
        self.should_fail = should_fail

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls += 1

        if self.should_fail:
            raise LLMGenerationError("Simulated failure for testing.")

        if "Extract structured data" in prompt:
            return (
                '{"conditions": ["Hypertension", "Diabetes"], '
                '"medications": [{"name": "Metformin", "dose": "500 mg", '
                '"frequency": "twice daily"}], '
                '"vitals": {"blood_pressure": "145/90"}, "allergies": []}'
            )
        if "Document context:" in prompt:
            return (
                "FROM THE DOCUMENTS:\n- (fake LLM) relevant facts here.\n\n"
                "EXPLANATION (general information, not from the documents):\n- None.\n\n"
                "LIMITATIONS:\n- None."
            )
        return "(fake LLM response)"


class BadJsonLLM(LLMProvider):
    """Always returns malformed JSON, for testing extraction retry/failure."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "this is not valid json"