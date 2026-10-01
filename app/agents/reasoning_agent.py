"""
Clinical Reasoning Agent: answers questions using ONLY retrieved context,
and clearly separates document facts from model-generated explanation.
"""

from app.llm.base import LLMProvider
from app.llm.llm_factory import get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = (
    "I could not find information about this in the uploaded documents."
)

SYSTEM_PROMPT = (
    "You are a careful healthcare information assistant for an educational "
    "demo project. You are NOT a doctor. You do not diagnose, prescribe, or "
    "change treatment. Answer using ONLY the provided document context. "
    "If the context does not contain the answer, say so plainly. Never "
    "invent facts, doses, or values."
)

REASONING_PROMPT = """Document context:
{context}

{history_block}Question: {question}

Respond in exactly this structure:

FROM THE DOCUMENTS:
- List the relevant facts, each citing its source like [Source: filename, chunk N].
- If nothing relevant is stated, write "Nothing relevant found."

EXPLANATION (general information, not from the documents):
- Briefly explain what these facts may mean in plain language.
- Do not diagnose or recommend treatment changes.
- Keep it short, or write "None." if not needed.

LIMITATIONS:
- State anything the documents do not cover that the question asks about.
"""


class ReasoningAgent:
    name = "reasoning_agent"

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_llm()

    def reason(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> str:

        # Hard guard: no context means no LLM call and no chance to hallucinate.
        if not context.strip():
            logger.info(
                f"{self.name}: empty context, returning not-found message"
            )
            return NOT_FOUND_MESSAGE

        history_block = ""

        if history:
            lines = [
                f"{m['role'].capitalize()}: {m['content']}"
                for m in history[-6:]
            ]

            history_block = (
                "Recent conversation:\n"
                + "\n".join(lines)
                + "\n\n"
            )

        prompt = REASONING_PROMPT.format(
            context=context,
            question=question,
            history_block=history_block,
        )

        return self._llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
        )
