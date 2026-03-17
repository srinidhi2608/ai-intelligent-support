"""
api/telemetry.py – REST endpoints for querying the synthetic telemetry data.

Exposes five read/write endpoints under the ``/api/v1`` prefix:

  GET  /api/v1/merchants/{merchant_id}
  GET  /api/v1/merchants/{merchant_id}/transactions
  GET  /api/v1/transactions/{transaction_id}
  GET  /api/v1/merchants/{merchant_id}/webhooks
  POST /api/v1/webhooks/{log_id}/retry

The ``DataLoader`` instance is accessed via ``request.app.state.data_loader``
which is populated during the application lifespan (see ``main.py``).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ──────────────────────────────────────────────────────────────────────────────


class MerchantResponse(BaseModel):
    """Profile of a single merchant."""

    merchant_id: str = Field(..., description="Stable unique merchant identifier")
    business_name: str = Field(..., description="Registered business name")
    mcc_code: str = Field(..., description="ISO 18245 Merchant Category Code")
    webhook_url: str = Field(..., description="HTTPS endpoint for payment events")


class TransactionResponse(BaseModel):
    """Details of a single payment transaction."""

    transaction_id: str
    merchant_id: str
    timestamp: str
    amount: float
    currency: str
    status: str
    decline_code: Optional[str] = None
    card_bin: Optional[str] = None


class WebhookLogResponse(BaseModel):
    """Delivery record for a single webhook notification."""

    log_id: str
    transaction_id: str
    timestamp: str
    event_type: str
    http_status: int
    delivery_attempts: int
    latency_ms: int


class TransactionDetailsResponse(BaseModel):
    """Full transaction details enriched with the associated webhook delivery log."""

    transaction_id: str
    merchant_id: str
    timestamp: str
    amount: float
    currency: str
    status: str
    decline_code: Optional[str] = None
    card_bin: Optional[str] = None
    webhook_log: Optional[WebhookLogResponse] = None


class RetryResponse(BaseModel):
    """Result of a simulated webhook retry action."""

    success: bool = Field(..., description="Whether the update was applied")
    log_id: str = Field(..., description="The log entry that was updated")
    new_http_status: int = Field(..., description="The HTTP status written")
    message: str = Field(..., description="Human-readable result message")


# ──────────────────────────────────────────────────────────────────────────────
# Dependency helper
# ──────────────────────────────────────────────────────────────────────────────


def _get_loader(request: Request):
    """
    Extract the ``DataLoader`` from ``app.state``.

    Raises a 503 if the data layer was never initialised (e.g. CSVs missing
    and the app started anyway).
    """
    loader = getattr(request.app.state, "data_loader", None)
    if loader is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Data layer is not available. "
                "Run `python data/telemetry_generator.py` to generate the "
                "CSV files, then restart the application."
            ),
        )
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/merchants/{merchant_id}",
    response_model=MerchantResponse,
    summary="Get merchant profile",
    responses={404: {"description": "Merchant not found"}},
)
def get_merchant(merchant_id: str, request: Request) -> MerchantResponse:
    """
    Return the profile for a single merchant.

    Parameters
    ----------
    merchant_id:
        The stable merchant ID (e.g. ``merchant_id_1``).

    Raises
    ------
    HTTPException(404):
        If no merchant with the given ID exists in the dataset.
    """
    loader = _get_loader(request)
    merchant = loader.get_merchant(merchant_id)
    if merchant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Merchant '{merchant_id}' not found.",
        )
    return MerchantResponse(**merchant)


@router.get(
    "/merchants/{merchant_id}/transactions",
    response_model=list[TransactionResponse],
    summary="List recent transactions for a merchant",
)
def get_merchant_transactions(
    merchant_id: str,
    request: Request,
    limit: int = Query(default=10, ge=1, le=500, description="Max rows to return"),
) -> list[TransactionResponse]:
    """
    Return the *limit* most-recent transactions for a merchant, newest first.

    Parameters
    ----------
    merchant_id:
        The merchant to query.
    limit:
        Number of transactions to return (default: 10, max: 500).

    Returns
    -------
    list[TransactionResponse]
        An empty list if the merchant exists but has no transactions.
    """
    loader = _get_loader(request)

    # Validate the merchant exists first so callers get a clear 404
    if loader.get_merchant(merchant_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Merchant '{merchant_id}' not found.",
        )

    rows = loader.get_recent_transactions(merchant_id, limit=limit)
    return [TransactionResponse(**row) for row in rows]


@router.get(
    "/transactions/{transaction_id}/details",
    response_model=TransactionDetailsResponse,
    summary="Get transaction details with associated webhook log",
    responses={404: {"description": "Transaction not found"}},
)
def get_transaction_with_webhook(
    transaction_id: str, request: Request
) -> TransactionDetailsResponse:
    """
    Return the full details of a single transaction, enriched with its
    associated webhook delivery log if one exists.

    This is the preferred endpoint for the AI agent because it provides both
    the financial status (including any ``decline_code``) AND the webhook
    delivery status (including the ``log_id`` needed to call the retry
    endpoint) in a single round-trip.

    Parameters
    ----------
    transaction_id:
        The unique transaction ID (e.g. ``TXN-00000001``).

    Raises
    ------
    HTTPException(404):
        If the transaction ID does not exist.
    """
    loader = _get_loader(request)
    txn = loader.get_transaction_details(transaction_id)
    if txn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{transaction_id}' not found.",
        )

    webhook_log = loader.get_webhook_log_for_transaction(transaction_id)
    webhook_model = WebhookLogResponse(**webhook_log) if webhook_log else None

    return TransactionDetailsResponse(
        **txn,
        webhook_log=webhook_model,
    )


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionResponse,
    summary="Get details for a single transaction",
    responses={404: {"description": "Transaction not found"}},
)
def get_transaction(transaction_id: str, request: Request) -> TransactionResponse:
    """
    Return the full details of a single transaction.

    Parameters
    ----------
    transaction_id:
        The unique transaction ID (e.g. ``TXN-00000001``).

    Raises
    ------
    HTTPException(404):
        If the transaction ID does not exist.
    """
    loader = _get_loader(request)
    txn = loader.get_transaction_details(transaction_id)
    if txn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{transaction_id}' not found.",
        )
    return TransactionResponse(**txn)


@router.get(
    "/merchants/{merchant_id}/webhooks",
    response_model=list[WebhookLogResponse],
    summary="List recent webhook delivery logs for a merchant",
)
def get_merchant_webhooks(
    merchant_id: str,
    request: Request,
    limit: int = Query(default=10, ge=1, le=500, description="Max rows to return"),
) -> list[WebhookLogResponse]:
    """
    Return the *limit* most-recent webhook delivery logs for a merchant.

    Internally this joins ``webhook_logs`` → ``transactions`` on
    ``transaction_id`` to resolve the merchant association.

    Parameters
    ----------
    merchant_id:
        The merchant to query.
    limit:
        Number of webhook log entries to return (default: 10, max: 500).

    Raises
    ------
    HTTPException(404):
        If the merchant does not exist.
    """
    loader = _get_loader(request)

    if loader.get_merchant(merchant_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Merchant '{merchant_id}' not found.",
        )

    logs = loader.get_webhook_logs_for_merchant(merchant_id, limit=limit)
    return [WebhookLogResponse(**log) for log in logs]


@router.post(
    "/webhooks/{log_id}/retry",
    response_model=RetryResponse,
    summary="Simulate an agentic webhook retry",
    responses={404: {"description": "Webhook log not found"}},
)
def retry_webhook(log_id: str, request: Request) -> RetryResponse:
    """
    Simulate an AI-agent action that retries a failed webhook delivery.

    Sets the ``http_status`` of the given webhook log entry to ``200``
    in memory, representing a successful re-delivery after the agent
    diagnosed and resolved the failure.

    Parameters
    ----------
    log_id:
        The ``log_id`` of the webhook entry to mark as retried
        (e.g. ``WH-00000001``).

    Raises
    ------
    HTTPException(404):
        If no webhook log with the given ``log_id`` exists.
    """
    loader = _get_loader(request)
    new_status = 200

    updated = loader.update_webhook_status(log_id, new_status)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Webhook log '{log_id}' not found.",
        )

    return RetryResponse(
        success=True,
        log_id=log_id,
        new_http_status=new_status,
        message=(
            f"Webhook log '{log_id}' has been marked as successfully "
            f"retried (http_status → {new_status})."
        ),
    )
