"""
Clinical Data Extraction Agent: converts free-text clinical notes into
validated, structured data.
"""

import re

from pydantic import ValidationError

from app.llm.base import LLMProvider
from app.llm.llm_factory import get_llm
from app.models.clinical import ClinicalExtraction
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical information extraction system. You extract ONLY "
    "information that is explicitly stated in the text. You never infer, "
    "guess, or add information. If something is not stated, use null or "
    "an empty list. You respond with valid JSON only, no commentary."
)

EXTRACTION_PROMPT = """Extract structured data from the clinical text below.

Return JSON in exactly this shape:
{{
  "conditions": ["..."],
  "medications": [{{"name": "...", "dose": "...", "frequency": "..."}}],
  "vitals": {{
    "blood_pressure": "...", "heart_rate": "...", "temperature": "...",
    "weight": "...", "oxygen_saturation": "..."
  }},
  "allergies": ["..."]
}}

Rules:
- Use null for any vital, dose, or frequency not stated.
- Use [] for any list with no stated items.
- Copy values as written (e.g. "500 mg", "145/90").

Clinical text:
\"\"\"
{text}
\"\"\"
"""


class ClinicalExtractionError(Exception):
    """Raised when structured extraction fails after retries."""
    pass


def _extract_json_block(raw: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences or add text; isolate the JSON."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return match.group(0) if match else raw


class ClinicalAgent:
    name = "clinical_agent"

    def __init__(self, llm: LLMProvider | None = None, max_attempts: int = 2) -> None:
        self._llm = llm or get_llm()
        self._max_attempts = max_attempts

    def extract(self, text: str) -> ClinicalExtraction:
        if not text or not text.strip():
            raise ClinicalExtractionError("No text provided for extraction.")

        prompt = EXTRACTION_PROMPT.format(text=text)
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            raw = self._llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
            try:
                return ClinicalExtraction.model_validate_json(_extract_json_block(raw))
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    f"{self.name}: invalid structured output "
                    f"(attempt {attempt}/{self._max_attempts})"
                )

        raise ClinicalExtractionError(
            f"Could not produce valid structured output: {last_error}"
        )