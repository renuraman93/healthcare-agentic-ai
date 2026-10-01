"""
Phase 7 sanity check: runs every route through the LangGraph workflow.

Usage:
    python test_phase7.py          -> FakeLLM (no Gemini quota used)
    python test_phase7.py real     -> real Gemini LLM (about 7 LLM calls)

Note: document search still calls the embedding API in both modes.
"""

import sys

from app.graph.workflow import HealthcareWorkflow
from tests.fakes import FakeLLM

PASTED_NOTE = (
    "Patient has a history of hypertension and diabetes. "
    "Current medication is Metformin 500 mg twice daily. "
    "Blood pressure is 145/90."
)

SCENARIOS = [
    # (label, session_id, question)
    ("Document question",      "s1", "What does the uploaded document say about hypertension?"),
    ("Follow-up (memory)",     "s1", "What medications are mentioned for it?"),
    ("Summarization",          "s2", "Give me a short summary of the uploaded notes"),
    ("Extraction (pasted)",    "s3", "Extract structured data from this note: " + PASTED_NOTE),
    ("General health",         "s4", "What is hypertension?"),
    ("Emergency wording",      "s5", "I have chest pain, what does the patient record say about my heart?"),
    ("Off-topic",              "s6", "What is the capital of France?"),
]


def main() -> None:
    use_real = len(sys.argv) > 1 and sys.argv[1].lower() == "real"

    if use_real:
        workflow = HealthcareWorkflow()
    else:
        print(">>> FAKE LLM MODE (no Gemini quota used)\n")
        workflow = HealthcareWorkflow(llm=FakeLLM())

    try:
        print("--- Graph (Mermaid) ---")
        print(workflow.graph.get_graph().draw_mermaid())
    except Exception as exc:
        print(f"(Could not render graph diagram: {type(exc).__name__})")

    for label, session_id, question in SCENARIOS:
        print(f"\n{'=' * 10} {label} {'=' * 10}")
        print(f"Q: {question[:90]}")
        result = workflow.run(question, session_id=session_id)
        print(f"agent={result.agent} | emergency={result.emergency_flag} | "
              f"sources={[(s.source, s.chunk_index) for s in result.sources]}")
        print("A:", result.answer[:350].replace("\n", "\n   "))


if __name__ == "__main__":
    main()