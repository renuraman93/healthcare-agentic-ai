"""
Unit tests for individual agents, using FakeLLM (no API calls).
"""

import pytest

from app.agents.clinical_agent import ClinicalAgent, ClinicalExtractionError
from app.agents.reasoning_agent import ReasoningAgent, NOT_FOUND_MESSAGE
from app.agents.response_agent import ResponseAgent, detect_emergency
from app.agents.summarization_agent import SummarizationAgent
from app.agents.supervisor_agent import SupervisorAgent, contextualize_query
from app.models.schemas import SourceRef
from tests.fakes import BadJsonLLM, FakeLLM


# ---- Supervisor / routing ----

@pytest.mark.parametrize("question,expected", [
    ("Extract structured data from this note", "clinical_extraction"),
    ("Please summarize the uploaded report", "summarization"),
    ("What does the uploaded document say about diabetes?", "document"),
    ("What is hypertension?", "general"),
    ("Explain how insulin works", "general"),
])
def test_rule_based_routing(question, expected):
    supervisor = SupervisorAgent(llm=None)
    assert supervisor.route(question) == expected


def test_routing_defaults_to_document_when_ambiguous():
    supervisor = SupervisorAgent(llm=None)
    # Deliberately vague text that matches none of the rule patterns
    assert supervisor.route("asdf qwerty zzz") == "document"


def test_contextualize_query_prepends_for_short_followup():
    history = [
        {"role": "user", "content": "What does the document say about hypertension?"},
        {"role": "assistant", "content": "..."},
    ]
    result = contextualize_query("What medications are mentioned for it?", history)
    assert "hypertension" in result


def test_contextualize_query_leaves_standalone_question_alone():
    history = [{"role": "user", "content": "What does the document say about hypertension?"}]
    long_question = "What does the uploaded document say about the patient's diabetes management plan?"
    assert contextualize_query(long_question, history) == long_question


# ---- Clinical extraction ----

def test_clinical_extraction_returns_valid_structure():
    agent = ClinicalAgent(llm=FakeLLM())
    result = agent.extract("Patient has hypertension. Metformin 500 mg twice daily.")
    assert "Hypertension" in result.conditions
    assert result.medications[0].name == "Metformin"


def test_clinical_extraction_empty_text_raises():
    agent = ClinicalAgent(llm=FakeLLM())
    with pytest.raises(ClinicalExtractionError):
        agent.extract("")


def test_clinical_extraction_bad_json_raises_after_retries():
    agent = ClinicalAgent(llm=BadJsonLLM(), max_attempts=2)
    with pytest.raises(ClinicalExtractionError):
        agent.extract("Patient has hypertension.")


# ---- Reasoning ----

def test_reasoning_empty_context_returns_not_found_without_llm_call():
    llm = FakeLLM()
    agent = ReasoningAgent(llm=llm)
    result = agent.reason("What medications?", context="")
    assert result == NOT_FOUND_MESSAGE
    assert llm.calls == 0  # confirms no LLM call was made


def test_reasoning_with_context_calls_llm():
    llm = FakeLLM()
    agent = ReasoningAgent(llm=llm)
    result = agent.reason("What medications?", context="[Source: doc.txt, chunk 0]\nMetformin 500mg")
    assert llm.calls == 1
    assert "FROM THE DOCUMENTS" in result


def test_reasoning_propagates_llm_failure(failing_llm):
    """
    This is the regression test for the fix in Step 2: reasoning_agent
    must NOT swallow LLM exceptions itself; the caller (workflow.py)
    is responsible for catching them.
    """
    from app.llm.base import LLMGenerationError
    agent = ReasoningAgent(llm=failing_llm)
    with pytest.raises(LLMGenerationError):
        agent.reason("What medications?", context="some context")


# ---- Summarization ----

def test_summarization_unknown_mode_raises():
    agent = SummarizationAgent(llm=FakeLLM())
    with pytest.raises(ValueError):
        agent.summarize("some text", mode="not_a_real_mode")


def test_summarization_empty_text_returns_message_without_llm_call():
    llm = FakeLLM()
    agent = SummarizationAgent(llm=llm)
    result = agent.summarize("", mode="short")
    assert "no text" in result.lower()
    assert llm.calls == 0


# ---- Response agent / emergency detection ----

@pytest.mark.parametrize("message,expected", [
    ("I have chest pain and can't breathe", True),
    ("What is metformin used for?", False),
    ("I'm having thoughts of self-harm", True),
    ("What are stroke symptoms?", True),  # matches "stroke" - intentionally over-sensitive
    ("", False),
])
def test_detect_emergency(message, expected):
    assert detect_emergency(message) == expected


def test_response_agent_includes_sources_and_disclaimer():
    responder = ResponseAgent()
    result = responder.compose(
        answer="Test answer",
        sources=[SourceRef(source="doc.txt", chunk_index=0, distance=0.1)],
        agent="reasoning_agent",
    )
    assert "Test answer" in result.answer
    assert "doc.txt" in result.answer
    assert "not medical advice" in result.answer.lower()
    assert result.agent == "reasoning_agent"


def test_response_agent_emergency_banner_appears_first():
    responder = ResponseAgent()
    result = responder.compose(
        answer="Answer text", sources=[], agent="reasoning_agent", emergency=True
    )
    assert result.answer.startswith("⚠️")
    assert result.emergency_flag is True