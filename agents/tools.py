"""
agents/tools.py – Python functions exposed as LangChain tools that agents can
                  call during their reasoning loop.

Each function is decorated with @tool so LangChain can auto-generate the
JSON schema that is sent to the LLM for function-calling.
"""

import json
import random

from langchain_core.tools import tool

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────────────────


@tool
def fetch_transaction_status(transaction_id: str) -> str:
    """
    Fetch the current status and details of a payment transaction.

    Use this tool whenever the merchant asks about a specific transaction,
    a payment failure, or a refund status.

    Args:
        transaction_id: The unique identifier of the transaction to look up.

    Returns:
        A JSON string containing transaction details such as status, amount,
        error code, and gateway response.
    """
    # TODO: replace with a real database / gateway API call
    # For now we return synthetic mock data so the agent can be tested
    # without live infrastructure.
    mock_statuses = ["SUCCESS", "FAILED", "PENDING", "REFUNDED"]
    mock_error_codes = [None, "INSUFFICIENT_FUNDS", "CARD_EXPIRED", "DO_NOT_HONOUR"]

    mock_response = {
        "transaction_id": transaction_id,
        "status": random.choice(mock_statuses),
        "amount": round(random.uniform(100.0, 50000.0), 2),
        "currency": "INR",
        "error_code": random.choice(mock_error_codes),
        "gateway": "mock-gateway-v1",
        "timestamp": "2024-01-15T10:30:00Z",
    }

    return json.dumps(mock_response)


@tool
def get_merchant_profile(merchant_id: str) -> str:
    """
    Retrieve KYB/KYC profile information for a given merchant.

    Use this tool when diagnosing onboarding issues or verifying merchant
    identity documents.

    Args:
        merchant_id: The unique identifier of the merchant.

    Returns:
        A JSON string containing the merchant's profile and KYB/KYC status.
    """
    # TODO: replace with a real merchant registry lookup
    mock_profile = {
        "merchant_id": merchant_id,
        "business_name": "Mock Business Ltd.",
        "kyb_status": "PENDING",   # PENDING | APPROVED | REJECTED
        "kyc_status": "APPROVED",
        "risk_tier": "MEDIUM",
        "registered_at": "2024-01-01T00:00:00Z",
    }

    return json.dumps(mock_profile)
