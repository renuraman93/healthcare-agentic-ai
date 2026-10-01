"""
General Healthcare Agent: answers general health-education questions that
are not about the uploaded documents. Output is always labeled so users can
tell it apart from document-grounded answers.
"""

from app.llm.base import LLMProvider
from app.llm.llm_factory import get_llm

SYSTEM_PROMPT = (
    "You are a healthcare education assistant for an educational demo "
    "project. Give general, widely accepted health information in plain "
    "language, in under 150 words. You are NOT a doctor: do not diagnose, "
    "do not recommend doses or treatment changes for an individual, and say "
    "when you are unsure. If the question is not about health or medicine, "
    "politely say you can only help with healthcare topics. Recommend "
    "consulting a clinician for personal medical decisions."
)

LABEL = "GENERAL INFORMATION (not from your uploaded documents):"


class GeneralAgent:
    name = "general_agent"

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_llm()

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        history_block = ""
        if history:
            lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]]
            history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        prompt = f"{history_block}Question: {question}"
        text = self._llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        return f"{LABEL}\n{text.strip()}"