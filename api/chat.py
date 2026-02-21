"""
api/chat.py – Merchant support chat endpoint.

Provides a conversational interface that routes merchant queries to the
SupportAgent which diagnoses issues and returns actionable responses.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Payload sent by the merchant to start / continue a conversation."""

    merchant_id: str = Field(..., description="Unique identifier of the merchant")
    message: str = Field(..., description="Natural-language query from the merchant")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to continue a previous conversation",
    )


class ChatResponse(BaseModel):
    """Agent's reply returned to the merchant."""

    session_id: str
    reply: str
    sources: list[str] = Field(
        default_factory=list,
        description="References or log IDs the agent used to formulate the reply",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message to the merchant support agent",
)
def send_message(request: ChatRequest) -> ChatResponse:
    """
    Forward the merchant's message to the SupportAgent and return its reply.

    In a production system this handler would:
      1. Load or create a LangChain conversation session.
      2. Pass the message to SupportAgent.run(message).
      3. Stream or return the final response.
    """
    # TODO: integrate with SupportAgent
    # from agents.support_agent import SupportAgent
    # agent = SupportAgent()
    # reply = agent.run(request.message)

    # Placeholder response until the agent is wired up
    placeholder_reply = (
        f"Hello merchant {request.merchant_id}! "
        "Your query has been received. The support agent is not yet wired up – "
        "please implement SupportAgent.run() in agents/support_agent.py."
    )

    return ChatResponse(
        session_id=request.session_id or "session-placeholder-001",
        reply=placeholder_reply,
        sources=[],
    )
