"""
POST /chat — the main conversational endpoint.
"""

from fastapi import APIRouter, HTTPException, Request

from app.models.api import ChatRequest, ChatResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    workflow = request.app.state.workflow

    try:
        result = workflow.run(payload.message, session_id=payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # workflow.run() already catches internal failures and returns a
        # friendly AgentResult; this is a last-resort guard for anything
        # truly unexpected (e.g. a bug), so the client still gets JSON,
        # never a raw traceback.
        logger.error(f"/chat unexpected failure: {type(exc).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        agent=result.agent,
        emergency_flag=result.emergency_flag,
    )