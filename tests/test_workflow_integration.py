"""
Full integration test using REAL Gemini calls. Skipped automatically
unless RUN_INTEGRATION=1 is set, since it consumes API quota.

Run explicitly with:
    $env:RUN_INTEGRATION = "1"
    pytest tests/test_workflow_integration.py -v
"""

import os

import pytest

from app.graph.workflow import HealthcareWorkflow

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Integration test uses real Gemini quota; set RUN_INTEGRATION=1 to run.",
)


def test_full_pipeline_document_question():
    workflow = HealthcareWorkflow()
    result = workflow.run(
        "What medications is the patient taking?", session_id="integration-test"
    )
    assert result.agent == "reasoning_agent"
    assert len(result.sources) > 0
    assert "metformin" in result.answer.lower() or "lisinopril" in result.answer.lower()


def test_full_pipeline_conversation_memory():
    workflow = HealthcareWorkflow()
    workflow.run("What does the document say about hypertension?", session_id="integration-test-2")
    result = workflow.run("What medications are mentioned for it?", session_id="integration-test-2")
    assert len(result.sources) > 0