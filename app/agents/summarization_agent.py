"""
Summarization Agent: summarizes medical text in several modes.
"""

from app.llm.base import LLMProvider
from app.llm.llm_factory import get_llm

SYSTEM_PROMPT = (
    "You are a medical text summarizer for an educational demo project. "
    "Summarize ONLY what is in the provided text. Do not add outside "
    "information, diagnoses, or recommendations. Preserve numbers, units, "
    "and drug names exactly as written."
)

MODE_INSTRUCTIONS = {
    "short": "Write a summary of at most 3 sentences.",
    "detailed": "Write a detailed summary in several paragraphs covering all major points.",
    "clinical": (
        "Write a clinical-style summary with these headings: "
        "History, Current Medications, Vitals, Plan. "
        "Write 'Not stated' under any heading with no information."
    ),
    "key_findings": "List the key findings as concise bullet points.",
}


class SummarizationAgent:
    name = "summarization_agent"
    modes = tuple(MODE_INSTRUCTIONS.keys())

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_llm()

    def summarize(self, text: str, mode: str = "short") -> str:
        if mode not in MODE_INSTRUCTIONS:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {self.modes}")
        if not text or not text.strip():
            return "There is no text to summarize."

        prompt = f"{MODE_INSTRUCTIONS[mode]}\n\nText:\n\"\"\"\n{text}\n\"\"\""
        return self._llm.generate(prompt, system_prompt=SYSTEM_PROMPT)