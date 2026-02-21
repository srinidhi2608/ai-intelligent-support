"""
api/webhooks.py – Mock FastAPI endpoints that receive payment lifecycle events.

These endpoints simulate a payment gateway callback system.  Real gateway
providers (Stripe, Razorpay, etc.) would POST to these URLs when a payment
event occurs (authorised, failed, refunded, etc.).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────


class PaymentEventPayload(BaseModel):
    """Schema for an incoming payment event webhook."""

    transaction_id: str = Field(..., description="Unique payment transaction ID")
    merchant_id: str = Field(..., description="Identifier of the merchant")
    amount: float = Field(..., gt=0, description="Transaction amount in the currency unit")
    currency: str = Field(default="INR", description="ISO 4217 currency code")
    status: str = Field(
        ...,
        description="Payment status: SUCCESS | FAILED | PENDING | REFUNDED",
    )
    error_code: str | None = Field(
        default=None,
        description="Gateway error code if the payment failed (e.g. INSUFFICIENT_FUNDS)",
    )
    metadata: dict | None = Field(
        default=None,
        description="Additional key-value pairs forwarded by the gateway",
    )


class WebhookAcknowledgement(BaseModel):
    """Standard acknowledgement returned for every webhook."""

    received: bool = True
    transaction_id: str
    message: str


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/payment",
    response_model=WebhookAcknowledgement,
    summary="Receive a payment lifecycle event",
)
def receive_payment_event(payload: PaymentEventPayload) -> WebhookAcknowledgement:
    """
    Accept a payment event from the gateway and echo it back as an
    acknowledgement.

    In a production system this handler would:
      1. Persist the event to a database / message queue.
      2. Trigger the SupportAgent to diagnose any failures.
      3. Update merchant dashboards in real time.
    """
    # TODO: persist the event and trigger downstream agents
    print(f"[WEBHOOK] Received payment event: {payload.model_dump()}")

    return WebhookAcknowledgement(
        received=True,
        transaction_id=payload.transaction_id,
        message=f"Payment event for transaction {payload.transaction_id} acknowledged.",
    )
