"""
LangGraph workflow for the healthcare agentic AI system.

Routes:
    document              -> document retrieval -> reasoning
    clinical_extraction   -> clinical data extraction
    summarization         -> document retrieval -> summarization
    general               -> general healthcare information

The workflow keeps conversation history using ConversationMemory and
returns an AgentResult for every request.
"""

from typing import Any
import uuid

from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.memory import ConversationMemory

from app.agents.supervisor_agent import (
    SupervisorAgent,
    contextualize_query,
)
from app.agents.document_agent import DocumentAgent
from app.agents.clinical_agent import ClinicalAgent
from app.agents.summarization_agent import SummarizationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.general_agent import GeneralAgent
from app.agents.response_agent import (
    ResponseAgent,
    detect_emergency,
)

from app.models.schemas import AgentResult
from app.utils.logger import get_logger


logger = get_logger(__name__)


class HealthcareWorkflow:
    """
    Main LangGraph workflow for the healthcare chatbot.
    """

    def __init__(
        self,
        llm=None,
        memory: ConversationMemory | None = None,
        retriever=None,
    ) -> None:

        self.memory = memory or ConversationMemory()

        # All agents receive the same LLM instance.
        # This is important for Phase 7 FakeLLM testing.
        self.supervisor = SupervisorAgent(llm=llm)
        self.document_agent = DocumentAgent(retriever=retriever)
        self.clinical_agent = ClinicalAgent(llm=llm)
        self.summarization_agent = SummarizationAgent(llm=llm)
        self.reasoning_agent = ReasoningAgent(llm=llm)
        self.general_agent = GeneralAgent(llm=llm)
        self.response_agent = ResponseAgent()

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(GraphState)

        # Common preparation
        graph.add_node("prepare", self._prepare)

        # Supervisor
        graph.add_node("supervisor", self._supervisor)

        # Route-specific nodes
        graph.add_node("document_search", self._document_search)
        graph.add_node("document_reasoning", self._document_reasoning)

        graph.add_node("clinical_extraction", self._clinical_extraction)

        graph.add_node("summarization_search", self._summarization_search)
        graph.add_node("summarization", self._summarization)

        graph.add_node("general", self._general)

        # Final response
        graph.add_node("finalize", self._finalize)

        # Entry
        graph.set_entry_point("prepare")

        # prepare -> supervisor
        graph.add_edge("prepare", "supervisor")

        # supervisor -> route
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "document": "document_search",
                "clinical_extraction": "clinical_extraction",
                "summarization": "summarization_search",
                "general": "general",
            },
        )

        # Document route
        graph.add_edge("document_search", "document_reasoning")
        graph.add_edge("document_reasoning", "finalize")

        # Clinical extraction route
        graph.add_edge("clinical_extraction", "finalize")

        # Summarization route
        graph.add_edge("summarization_search", "summarization")
        graph.add_edge("summarization", "finalize")

        # General route
        graph.add_edge("general", "finalize")

        # Final node
        graph.add_edge("finalize", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node 1: prepare
    # ------------------------------------------------------------------

    def _prepare(self, state: GraphState) -> dict[str, Any]:
        """
        Load conversation history and detect emergency wording.
        """

        question = state.get("question", "").strip()
        request_id = state.get("request_id", "")
        history = state.get("history", [])

        emergency = detect_emergency(question)

        logger.info(
            f"Preparing request {request_id or '<no-id>'}; "
            f"emergency={emergency}"
        )

        return {
            "question": question,
            "history": history,
            "emergency": emergency,
        }

    # ------------------------------------------------------------------
    # Node 2: supervisor
    # ------------------------------------------------------------------

    def _supervisor(self, state: GraphState) -> dict[str, Any]:
        """
        Determine which specialized agent should handle the request.
        """

        question = state["question"]
        history = state.get("history", [])

        # Resolve short follow-up questions such as:
        # "What medications are mentioned for it?"
        contextual_question = contextualize_query(
            question,
            history,
        )

        route = self.supervisor.route(contextual_question)

        logger.info(
            f"Supervisor route: {route} | "
            f"question_len={len(contextual_question)}"
        )

        return {
            "question": contextual_question,
            "retrieval_query": contextual_question,
            "route": route,
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_supervisor(self, state: GraphState) -> str:
        route = state.get("route", "document")

        allowed = {
            "document",
            "clinical_extraction",
            "summarization",
            "general",
        }

        if route not in allowed:
            logger.warning(
                f"Unknown route '{route}'. Falling back to document."
            )
            return "document"

        return route

    # ------------------------------------------------------------------
    # Document search
    # ------------------------------------------------------------------

    def _document_search(self, state: GraphState) -> dict[str, Any]:
        """
        Retrieve relevant document chunks.
        """

        query = state.get(
            "retrieval_query",
            state.get("question", ""),
        )

        result = self.document_agent.search(
            query=query,
            apply_threshold=True,
        )

        logger.info(
            f"Document search: found={result.found}, "
            f"sources={len(result.sources)}"
        )

        return {
            "doc_found": result.found,
            "context": result.context,
            "sources": result.sources,
        }

    # ------------------------------------------------------------------
    # Document reasoning
    # ------------------------------------------------------------------

    def _document_reasoning(self, state: GraphState) -> dict[str, Any]:
        """
        Generate an answer grounded in retrieved document context.
        """
        try:
            answer = self.reasoning_agent.reason(
                question=state["question"],
                context=state.get("context", ""),
                history=state.get("history", []),
            )
        except Exception as exc:
            logger.error(
                f"Document reasoning failed: {type(exc).__name__}"
            )
            return {
                "answer": (
                    "I was able to retrieve relevant information from the "
                    "healthcare documents, but the language model is "
                    "temporarily unavailable. Please try again."
                ),
                "agent": self.reasoning_agent.name,
                "error": type(exc).__name__,
                "failed_node": "document_reasoning",
            }

        return {
            "answer": answer,
            "agent": self.reasoning_agent.name,
        }

    # ------------------------------------------------------------------
    # Clinical extraction
    # ------------------------------------------------------------------

    def _clinical_extraction(self, state: GraphState) -> dict[str, Any]:
        """
        Extract structured clinical information.

        The extraction request normally contains the clinical text
        after the phrase 'Extract structured data from this note:'.
        """

        question = state["question"]

        text = question

        marker = "Extract structured data from this note:"

        if marker.lower() in question.lower():
            index = question.lower().find(marker.lower())
            text = question[index + len(marker):].strip()

        try:
            extraction = self.clinical_agent.extract(text)

            # Convert Pydantic model into a dictionary for GraphState.
            extraction_dict = extraction.model_dump()

            # User-facing representation.
            answer = extraction.model_dump_json(indent=2)

            return {
                "extraction": extraction_dict,
                "answer": answer,
                "agent": self.clinical_agent.name,
            }

        except Exception as exc:
            logger.error(
                f"Clinical extraction failed: {type(exc).__name__}: {exc}"
            )

            return {
                "answer": (
                    "I could not extract the clinical information "
                    "into the required structured format."
                ),
                "agent": self.clinical_agent.name,
                "error": str(exc),
                "failed_node": "clinical_extraction",
            }

    # ------------------------------------------------------------------
    # Summarization search
    # ------------------------------------------------------------------

    def _summarization_search(self, state: GraphState) -> dict[str, Any]:
        """
        Retrieve document content for summarization.

        Threshold filtering is intentionally disabled because
        summarization should summarize the available document rather
        than reject chunks based on the normal Q&A relevance threshold.
        """

        query = state.get(
            "retrieval_query",
            state.get("question", ""),
        )

        result = self.document_agent.search(
            query=query,
            apply_threshold=False,
        )

        logger.info(
            f"Summarization search: found={result.found}, "
            f"sources={len(result.sources)}"
        )

        return {
            "doc_found": result.found,
            "context": result.context,
            "sources": result.sources,
        }

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarization(self, state: GraphState) -> dict[str, Any]:
        """
        Summarize retrieved document context.
        """

        context = state.get("context", "")

        answer = self.summarization_agent.summarize(
            text=context,
            mode="short",
        )

        return {
            "answer": answer,
            "agent": self.summarization_agent.name,
        }

    # ------------------------------------------------------------------
    # General healthcare question
    # ------------------------------------------------------------------

    def _general(self, state: GraphState) -> dict[str, Any]:
        """
        Answer general healthcare questions.
        """

        answer = self.general_agent.answer(
            question=state["question"],
            history=state.get("history", []),
        )

        return {
            "answer": answer,
            "agent": self.general_agent.name,
        }

    # ------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------

    def _finalize(self, state: GraphState) -> dict[str, Any]:
        """
        Convert the internal graph state into the final AgentResult.
        """

        answer = state.get(
            "answer",
            "I could not generate a response.",
        )

        sources = state.get("sources", [])

        agent = state.get(
            "agent",
            "healthcare_workflow",
        )

        emergency = state.get("emergency", False)

        result = self.response_agent.compose(
            answer=answer,
            sources=sources,
            agent=agent,
            emergency=emergency,
        )

        return {
            "result": result,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        session_id: str = "default",
    ) -> AgentResult:
        """
        Run one user question through the complete workflow.

        Conversation history is loaded before execution and updated
        after the final response.
        """

        if not question or not question.strip():
            return AgentResult(
                answer="Please provide a question.",
                sources=[],
                agent="healthcare_workflow",
                emergency_flag=False,
            )

        # Get previous conversation for this session.
        history = self.memory.get(session_id)
        request_id = uuid.uuid4().hex[:8]

        initial_state: GraphState = {
            "request_id": request_id,
            "question": question.strip(),
            "history": history,
        }

        try:
            final_state = self.graph.invoke(initial_state)

            result = final_state.get("result")

            if not isinstance(result, AgentResult):
                # Defensive fallback in case LangGraph returns
                # a serialized/dict representation.
                if isinstance(result, dict):
                    result = AgentResult.model_validate(result)
                else:
                    raise RuntimeError(
                        "Workflow completed without an AgentResult."
                    )

        except Exception as exc:
            logger.exception(
                f"Healthcare workflow failed: {type(exc).__name__}: {exc}"
            )

            result = AgentResult(
                answer=(
                    "I’m sorry, but I could not complete the request "
                    "because of a temporary workflow error."
                ),
                sources=[],
                agent="healthcare_workflow",
                emergency_flag=detect_emergency(question),
            )

        # Save conversation after completion.
        self.memory.add(
            session_id,
            "user",
            question.strip(),
        )

        self.memory.add(
            session_id,
            "assistant",
            result.answer,
        )

        return result