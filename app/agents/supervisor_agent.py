"""
Supervisor Agent: decides which specialist handles a query.

Strategy: cheap deterministic rules first. Only if the rules are ambiguous
AND an LLM was provided do we spend an LLM call on classification.
Otherwise we default to the document route, which is the safest choice
because it is grounded in retrieved evidence.
"""

import re

from app.llm.base import LLMProvider
from app.utils.logger import get_logger

logger = get_logger(__name__)

ROUTES = ("clinical_extraction", "summarization", "document", "general")

_EXTRACTION_RE = re.compile(r"\b(extract|structured|json)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(
    r"\b(summari[sz]e|summary|key findings|key points|overview|recap)\b",
    re.IGNORECASE,
)
_DOC_RE = re.compile(
    r"\b(documents?|uploaded|report|notes?|records?|files?|patient|mentioned|according to)\b",
    re.IGNORECASE,
)
_GENERAL_RE = re.compile(
    r"^\s*(what (is|are|does)|explain|define|how (does|do)|why (is|do|does)|tell me about)\b",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"\b(it|its|that|this|those|these|them|they|he|she|his|her)\b", re.IGNORECASE
)

ROUTER_SYSTEM = "You classify user questions. Reply with exactly one label and nothing else."
ROUTER_PROMPT = """Classify this question into exactly one label:
- document: asks about the content of uploaded medical documents
- clinical_extraction: asks to extract structured data (conditions, medications, vitals)
- summarization: asks to summarize something
- general: a general health-knowledge question not about the documents

Question: {question}
Label:"""


def contextualize_query(question: str, history: list[dict]) -> str:
    """
    Cheap follow-up handling: if the question is short or uses pronouns
    ("What medications are mentioned for it?"), prepend the previous user
    message so retrieval has the missing context. An LLM-based query
    rewrite is more accurate but costs an extra call.
    """
    if not history:
        return question
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), None
    )
    if last_user and (len(question.split()) < 5 or _FOLLOWUP_RE.search(question)):
        return f"{last_user} {question}"
    return question


class SupervisorAgent:
    name = "supervisor_agent"

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    @staticmethod
    def classify_by_rules(question: str) -> str | None:
        """Returns a route, or None if the rules can't decide."""
        if _EXTRACTION_RE.search(question):
            return "clinical_extraction"
        if _SUMMARY_RE.search(question):
            return "summarization"
        if _DOC_RE.search(question):
            return "document"
        if _GENERAL_RE.search(question):
            return "general"
        return None

    def route(self, question: str) -> str:
        rule_route = self.classify_by_rules(question)
        if rule_route:
            return rule_route

        if self._llm is not None:
            try:
                reply = self._llm.generate(
                    ROUTER_PROMPT.format(question=question),
                    system_prompt=ROUTER_SYSTEM,
                ).lower()
                for candidate in ROUTES:
                    if candidate in reply:
                        return candidate
            except Exception as exc:
                logger.warning(f"LLM routing failed ({type(exc).__name__}); defaulting to document")

        return "document"