"""
Runs the three untested summary modes with a pause between calls
to stay under free-tier rate limits.
"""

import time

from app.agents.summarization_agent import SummarizationAgent
from app.llm.llm_factory import get_llm

SAMPLE_TEXT = (
    "Patient has a history of hypertension and diabetes. "
    "Current medication is Metformin 500 mg twice daily. "
    "Blood pressure is 145/90."
)


def main() -> None:
    summarizer = SummarizationAgent(get_llm())
    for mode in ("detailed", "clinical", "key_findings"):
        print(f"\n[{mode}]")
        print(summarizer.summarize(SAMPLE_TEXT, mode))
        time.sleep(15)


if __name__ == "__main__":
    main()