
"""
Phase 6 sanity check:
Exercises every agent individually.

Tests:
1. Document search + reasoning + response
2. Irrelevant question / RAG relevance
3. Clinical extraction
4. Summarization
5. Emergency detection
"""

from app.agents.document_agent import DocumentAgent
from app.agents.clinical_agent import ClinicalAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.summarization_agent import SummarizationAgent
from app.agents.response_agent import ResponseAgent, detect_emergency
from app.llm.llm_factory import get_llm


# =========================================================
# Sample healthcare document
# =========================================================

SAMPLE_TEXT = (
    "Patient has a history of hypertension and diabetes. "
    "Current medication is Metformin 500 mg twice daily. "
    "Blood pressure is 145/90."
)


# =========================================================
# Helper function
# =========================================================

def section(title: str) -> None:
    """Print a formatted section heading."""
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# =========================================================
# Main Phase 6 test
# =========================================================

def main() -> None:
    """Run all Phase 6 agent tests."""

    print("\nStarting Phase 6 Agent Tests...")

    # -----------------------------------------------------
    # Create one shared LLM instance
    # -----------------------------------------------------

    llm = get_llm()

    # -----------------------------------------------------
    # Initialize agents
    # -----------------------------------------------------

    doc_agent = DocumentAgent()
    clinical = ClinicalAgent(llm)
    reasoning = ReasoningAgent(llm)
    summarizer = SummarizationAgent(llm)
    responder = ResponseAgent()

    print("All agents initialized successfully.")

    # =====================================================
    # 1. DOCUMENT SEARCH + REASONING + RESPONSE
    # =====================================================

    section(
        "1. Document search + reasoning + response "
        "(relevant question)"
    )

    question = "What medications is the patient taking?"

    try:
        result = doc_agent.search(question)

        print(
            "Found:",
            result.found,
            "| Sources:",
            result.sources,
        )

        if result.found:
            answer = reasoning.reason(
                question,
                result.context,
            )

            final = responder.compose(
                answer,
                result.sources,
                agent="document_agent",
            )

            print("\nFinal Answer:")
            print(final.answer)

        else:
            print(
                "No relevant information found "
                "in the uploaded documents."
            )

    except Exception as e:
        print("Document search/reasoning failed:")
        print(e)

    # =====================================================
    # 2. IRRELEVANT QUESTION / RAG RELEVANCE
    # =====================================================

    section(
        "2. Irrelevant question "
        "(should say not found)"
    )

    question2 = "What is the capital of France?"

    try:
        result2 = doc_agent.search(question2)

        print("Found:", result2.found)
        print("Sources:", result2.sources)

        # -------------------------------------------------
        # Print vector distances.
        #
        # This helps choose the correct
        # RELEVANCE_MAX_DISTANCE in .env
        # -------------------------------------------------

        if result2.sources:
            print("\nRetrieved distances:")

            for source in result2.sources:
                print(
                    f"Source: {source.source} | "
                    f"Chunk: {source.chunk_index} | "
                    f"Distance: {source.distance}"
                )

        if result2.found:
            answer2 = reasoning.reason(
                question2,
                result2.context,
            )

            final2 = responder.compose(
                answer2,
                result2.sources,
                agent="document_agent",
            )

            print("\nAnswer:")
            print(final2.answer)

        else:
            print(
                "\nCorrectly identified as "
                "not found in the documents."
            )

    except Exception as e:
        print("Irrelevant-question test failed:")
        print(e)

    # =====================================================
    # 3. CLINICAL EXTRACTION
    # =====================================================

    section("3. Clinical extraction")

    try:
        extraction = clinical.extract(SAMPLE_TEXT)

        print(
            extraction.model_dump_json(
                indent=2
            )
        )

    except Exception as e:
        print("Clinical extraction failed:")
        print(e)

    # =====================================================
    # 4. SUMMARIZATION
    # =====================================================

    section("4. Summarization (all modes)")

    for mode in summarizer.modes:
        print(f"\n[{mode}]")

        try:
            result = summarizer.summarize(
                SAMPLE_TEXT,
                mode,
            )

            print(result)

        except Exception as e:
            print(
                f"Summarization unavailable "
                f"for mode '{mode}':"
            )

            print(e)

            print(
                "Continuing with the remaining "
                "Phase 6 tests..."
            )

    # =====================================================
    # 5. EMERGENCY DETECTION
    # =====================================================

    section("5. Emergency detection")

    messages = [
        "I have chest pain and can't breathe",
        "What is metformin used for?",
    ]

    for msg in messages:
        try:
            emergency = detect_emergency(msg)

            print(
                f"{msg!r} -> "
                f"emergency: {emergency}"
            )

        except Exception as e:
            print(
                f"Emergency detection failed "
                f"for {msg!r}: {e}"
            )

    # =====================================================
    # PHASE 6 COMPLETE
    # =====================================================

    print("\n" + "=" * 60)
    print("PHASE 6 TEST COMPLETED")
    print("=" * 60)


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()
