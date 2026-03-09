"""
agents/agent_tools.py – Agentic Tools for the Intelligent Merchant Support Agent.

This module defines the three LangChain ``@tool``-decorated functions that the
LLM agent calls during its reasoning loop:

1. ``fetch_transaction_logs``  – Diagnostic tool: retrieves live transaction
   details and webhook delivery status from the local FastAPI gateway.

2. ``retry_failed_webhook``    – Remediation tool: triggers an agentic retry of
   a failed webhook delivery via the FastAPI gateway.

3. ``search_knowledge_base``   – RAG tool: queries the local Chroma vector store
   for relevant FinTech documentation (decline codes, webhook rules, payout
   schedules).

All network calls point to ``http://localhost:8000`` — the FastAPI mock gateway
started with ``uvicorn main:app``.  The RAG tool uses the persisted Chroma
store built by ``python rag_setup.py``.

The exported list ``merchant_support_tools`` contains all three tools so they
can be bound to an LLM in a single line::

    from agents.agent_tools import merchant_support_tools
    llm_with_tools = chat_llm.bind_tools(merchant_support_tools)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain_core.tools import tool

from rag_setup import get_retriever

logger = logging.getLogger(__name__)

# ── Gateway base URL – override via environment if needed ─────────────────────
_GATEWAY_BASE_URL = "http://localhost:8000"

# ── HTTP request timeout (seconds) ───────────────────────────────────────────
_REQUEST_TIMEOUT = 10


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1 – Diagnostic Tool
# ──────────────────────────────────────────────────────────────────────────────


@tool
def fetch_transaction_logs(transaction_id: str) -> str:
    """
    Fetch the live transaction details and webhook logs for a specific
    transaction_id.  It returns the financial status, decline codes, and
    webhook delivery status in JSON format.

    Use this tool as the **first step** whenever a merchant reports a payment
    issue, a mysterious decline, or a webhook delivery failure.  The returned
    JSON contains everything you need to diagnose the problem:

    * ``transaction_id``  – The identifier you queried.
    * ``merchant_id``     – Which merchant the transaction belongs to.
    * ``timestamp``       – When the transaction occurred (ISO-8601 UTC).
    * ``amount``          – Transaction amount in the stated currency.
    * ``currency``        – Three-letter ISO 4217 currency code (e.g. ``INR``).
    * ``status``          – ``SUCCESS`` or ``DECLINED``.
    * ``decline_code``    – ISO 8583 decline reason (e.g. ``93_Risk_Block``,
                            ``51_Insufficient_Funds``).  ``null`` on success.
    * ``card_bin``        – First six digits of the card number (BIN).

    After calling this tool, if the ``decline_code`` is not obvious, use
    ``search_knowledge_base`` to look up its meaning.  If the associated
    webhook log shows a 5xx error, use ``retry_failed_webhook`` to remediate.

    Args:
        transaction_id: The unique transaction identifier to look up
            (e.g. ``TXN-00000042``).

    Returns:
        A JSON string containing full transaction details on success, or a
        plain-English error message if the transaction is not found or the
        gateway is unreachable.
    """
    url = f"{_GATEWAY_BASE_URL}/api/v1/transactions/{transaction_id}"
    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.text  # already a JSON string
        elif response.status_code == 404:
            return (
                f"Transaction '{transaction_id}' was not found in the gateway. "
                "Please verify the transaction ID and try again."
            )
        elif response.status_code == 503:
            return (
                "The data layer is unavailable. "
                "Run `python data/telemetry_generator.py` to generate the CSV "
                "files, then restart the FastAPI server."
            )
        else:
            # Unexpected status – surface the raw response for diagnostics
            return (
                f"Unexpected response from the gateway "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            )
    except requests.exceptions.ConnectionError:
        return (
            f"Could not connect to the payment gateway at {_GATEWAY_BASE_URL}. "
            "Ensure the FastAPI server is running: `uvicorn main:app --reload`."
        )
    except requests.exceptions.Timeout:
        return (
            f"The gateway request timed out after {_REQUEST_TIMEOUT} seconds. "
            "The server may be overloaded — please retry."
        )
    except requests.exceptions.RequestException as exc:
        logger.exception("Unexpected error in fetch_transaction_logs")
        return f"An unexpected error occurred while fetching transaction data: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2 – Remediation Tool
# ──────────────────────────────────────────────────────────────────────────────


@tool
def retry_failed_webhook(log_id: str) -> str:
    """
    Trigger a retry of a previously failed webhook delivery identified by
    log_id.  Use this tool ONLY when a merchant's webhook log shows a 500-level
    HTTP error or a timeout.

    **When to use:**

    * The webhook log's ``http_status`` is ``500``, ``502``, ``503``, or
      ``504``.
    * The merchant reports that their system did not receive the payment
      notification despite the transaction succeeding.
    * A previous ``fetch_transaction_logs`` call revealed a failed webhook
      entry.

    **When NOT to use:**

    * The ``http_status`` is ``401`` (Unauthorized) — the merchant's webhook
      secret is likely wrong; instruct them to rotate it in their dashboard
      instead.
    * The ``http_status`` is ``400`` (Bad Request) — this is a client-side
      payload error that a retry will not fix.
    * The transaction itself ``DECLINED`` — there is nothing to deliver.

    This tool simulates an AI-agent remediation action by calling the retry
    endpoint on the gateway, which marks the webhook log entry as successfully
    re-delivered (``http_status → 200``).

    Args:
        log_id: The unique webhook log identifier to retry
            (e.g. ``WH-00000007``).

    Returns:
        A plain-English confirmation string on success, or a diagnostic error
        message explaining why the retry could not be completed.
    """
    url = f"{_GATEWAY_BASE_URL}/api/v1/webhooks/{log_id}/retry"
    try:
        response = requests.post(url, timeout=_REQUEST_TIMEOUT)
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            return (
                f"Webhook retry succeeded for log '{log_id}'. "
                f"New HTTP status: {data.get('new_http_status', 200)}. "
                f"Message: {data.get('message', 'Re-delivery recorded.')}"
            )
        elif response.status_code == 404:
            return (
                f"Webhook log '{log_id}' was not found. "
                "Please verify the log_id (e.g. WH-00000007) and retry."
            )
        elif response.status_code == 503:
            return (
                "The data layer is unavailable — cannot perform the retry. "
                "Run `python data/telemetry_generator.py` first."
            )
        else:
            return (
                f"Webhook retry failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
    except requests.exceptions.ConnectionError:
        return (
            f"Could not connect to the payment gateway at {_GATEWAY_BASE_URL}. "
            "Ensure the FastAPI server is running: `uvicorn main:app --reload`."
        )
    except requests.exceptions.Timeout:
        return (
            f"The retry request timed out after {_REQUEST_TIMEOUT} seconds. "
            "Please retry the operation."
        )
    except requests.exceptions.RequestException as exc:
        logger.exception("Unexpected error in retry_failed_webhook")
        return f"An unexpected error occurred while retrying the webhook: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3 – Knowledge Base Tool (RAG)
# ──────────────────────────────────────────────────────────────────────────────


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal FinTech documentation knowledge base for authoritative
    answers on decline codes, webhook integration rules, and payout schedules.

    Always use this tool to look up the meaning of specific decline codes
    (like Code 93 or Code 51), webhook integration rules, or payout schedules
    before explaining them to the merchant.  Do NOT guess or hallucinate the
    meaning of a decline code — retrieve it from this knowledge base first.

    **Topics covered:**

    * **Decline codes** – ISO 8583 codes 05, 14, 51, 54, 57, 91, 93 with full
      descriptions, root causes, and recommended merchant actions.
    * **Webhook integration** – Expected HTTP response codes (200, 401, 500,
      504), retry policy (3 attempts with exponential back-off), and how to
      secure webhook endpoints with HMAC-SHA256 signatures.
    * **Payout schedules** – T+1 and T+2 settlement cycles, cutoff times,
      how bank holidays affect the payout date, and what to do if a payout is
      delayed.

    **Example queries:**

    * ``"What does decline code 93 mean?"``
    * ``"Why is my webhook returning 401?"``
    * ``"When will T+2 settlement arrive if the settlement date is a holiday?"``

    Args:
        query: A natural-language question about FinTech operations, decline
            codes, webhooks, or payout rules.

    Returns:
        A concatenated string of the most relevant documentation passages
        retrieved from the knowledge base, or an error message if the KB is
        not initialized.
    """
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return (
                "No relevant documentation was found for your query. "
                "Try rephrasing or run `python rag_setup.py` to rebuild the "
                "knowledge base."
            )
        # Concatenate retrieved chunks with clear separators so the LLM can
        # identify where each passage begins and ends.
        sections = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            sections.append(
                f"[Source {i}: {source}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(sections)
    except FileNotFoundError:
        return (
            "The knowledge base has not been initialized yet. "
            "Run `python rag_setup.py` from the project root to build it, "
            "then try again."
        )
    except Exception as exc:
        logger.exception("Unexpected error in search_knowledge_base")
        return (
            f"An unexpected error occurred while querying the knowledge base: "
            f"{exc}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tool export list
# ──────────────────────────────────────────────────────────────────────────────

# Import this list in your agent file and pass it to ``llm.bind_tools()``:
#
#   from agents.agent_tools import merchant_support_tools
#   llm_with_tools = chat_llm.bind_tools(merchant_support_tools)
#
merchant_support_tools = [
    fetch_transaction_logs,
    retry_failed_webhook,
    search_knowledge_base,
]
