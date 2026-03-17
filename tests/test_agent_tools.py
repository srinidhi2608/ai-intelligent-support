"""
tests/test_agent_tools.py – Unit tests for agents/agent_tools.py.

Strategy
--------
* All three tools make external calls (HTTP to the FastAPI gateway or Chroma/
  HuggingFace for RAG).  Every such call is patched so the suite runs fully
  offline and in milliseconds.
* We test:
  - Happy paths (200 / success responses).
  - All distinct error branches (404, 503, unexpected status, ConnectionError,
    Timeout, generic RequestException, FileNotFoundError, generic Exception).
  - Tool metadata (name, description exist, tools are LangChain-callable).
  - The ``merchant_support_tools`` export list.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.agent_tools import (
    fetch_merchant_diagnostics,
    fetch_transaction_logs,
    merchant_support_tools,
    retry_failed_webhook,
    search_knowledge_base,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_response(status_code: int, text: str = "", json_data: dict | None = None):
    """Build a lightweight mock that imitates a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Tool metadata
# ──────────────────────────────────────────────────────────────────────────────


class TestToolMetadata:
    def test_fetch_transaction_logs_has_name(self):
        assert fetch_transaction_logs.name == "fetch_transaction_logs"

    def test_retry_failed_webhook_has_name(self):
        assert retry_failed_webhook.name == "retry_failed_webhook"

    def test_search_knowledge_base_has_name(self):
        assert search_knowledge_base.name == "search_knowledge_base"

    def test_all_tools_have_non_empty_description(self):
        for t in merchant_support_tools:
            assert t.description and len(t.description) > 20

    def test_merchant_support_tools_length(self):
        assert len(merchant_support_tools) == 4

    def test_merchant_support_tools_order(self):
        names = [t.name for t in merchant_support_tools]
        assert names == [
            "fetch_transaction_logs",
            "retry_failed_webhook",
            "search_knowledge_base",
            "fetch_merchant_diagnostics",
        ]

    def test_tools_are_langchain_callable(self):
        """Each tool must have an .invoke() method (LangChain BaseTool)."""
        for t in merchant_support_tools:
            assert callable(getattr(t, "invoke", None))


# ──────────────────────────────────────────────────────────────────────────────
# fetch_transaction_logs
# ──────────────────────────────────────────────────────────────────────────────


class TestFetchTransactionLogs:
    _TXN_JSON = '{"transaction_id": "TXN-001", "status": "DECLINED"}'

    def test_200_returns_json_body(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=self._TXN_JSON)
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert "TXN-001" in result
        assert "DECLINED" in result

    def test_url_contains_transaction_id(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=self._TXN_JSON)
            fetch_transaction_logs.invoke({"transaction_id": "TXN-XYZ"})
        called_url = mock_get.call_args[0][0]
        assert "TXN-XYZ" in called_url

    def test_url_uses_details_endpoint(self):
        """fetch_transaction_logs must use the /details endpoint to include webhook data."""
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=self._TXN_JSON)
            fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        called_url = mock_get.call_args[0][0]
        assert "details" in called_url

    def test_404_returns_not_found_message(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404, text="Not Found")
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-MISSING"})
        assert "not found" in result.lower()
        assert "TXN-MISSING" in result

    def test_503_returns_data_layer_message(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(503, text="Service Unavailable")
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert "data layer" in result.lower() or "unavailable" in result.lower()

    def test_unexpected_status_surfaces_code(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(429, text="Rate Limited")
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert "429" in result

    def test_connection_error_returns_guidance(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = req_lib.exceptions.ConnectionError("refused")
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert "connect" in result.lower() or "gateway" in result.lower()

    def test_timeout_returns_timeout_message(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = req_lib.exceptions.Timeout("timed out")
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert "API_ERROR" in result or "gateway" in result.lower() or "offline" in result.lower()

    def test_generic_request_exception_returns_error(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = req_lib.exceptions.RequestException("boom")
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert "API_ERROR" in result or "error" in result.lower() or "gateway" in result.lower()

    def test_returns_string(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=self._TXN_JSON)
            result = fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        assert isinstance(result, str)

    def test_timeout_kwarg_is_set(self):
        """The GET call must use an explicit timeout to avoid hanging."""
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=self._TXN_JSON)
            fetch_transaction_logs.invoke({"transaction_id": "TXN-001"})
        _, kwargs = mock_get.call_args
        assert "timeout" in kwargs


# ──────────────────────────────────────────────────────────────────────────────
# retry_failed_webhook
# ──────────────────────────────────────────────────────────────────────────────


class TestRetryFailedWebhook:
    _RETRY_JSON = {
        "success": True,
        "log_id": "WH-001",
        "new_http_status": 200,
        "message": "Webhook log 'WH-001' marked as retried.",
    }

    def test_200_returns_success_message(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                200, text="", json_data=self._RETRY_JSON
            )
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "succeeded" in result.lower() or "success" in result.lower()

    def test_200_includes_log_id(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                200, text="", json_data=self._RETRY_JSON
            )
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "WH-001" in result

    def test_url_contains_log_id(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                200, text="", json_data=self._RETRY_JSON
            )
            retry_failed_webhook.invoke({"log_id": "WH-XYZ"})
        called_url = mock_post.call_args[0][0]
        assert "WH-XYZ" in called_url

    def test_url_contains_retry_segment(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                200, text="", json_data=self._RETRY_JSON
            )
            retry_failed_webhook.invoke({"log_id": "WH-001"})
        called_url = mock_post.call_args[0][0]
        assert "retry" in called_url

    def test_404_returns_not_found_message(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(404, text="Not Found")
            result = retry_failed_webhook.invoke({"log_id": "WH-MISSING"})
        assert "not found" in result.lower()
        assert "WH-MISSING" in result

    def test_503_returns_data_layer_message(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(503)
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "data layer" in result.lower() or "unavailable" in result.lower()

    def test_unexpected_status_surfaces_code(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(422, text="Unprocessable")
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "422" in result

    def test_connection_error_returns_guidance(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.ConnectionError("refused")
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "connect" in result.lower() or "gateway" in result.lower()

    def test_timeout_returns_timeout_message(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.Timeout("timed out")
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "API_ERROR" in result or "gateway" in result.lower() or "offline" in result.lower()

    def test_generic_request_exception_returns_error(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.RequestException("boom")
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert "API_ERROR" in result or "error" in result.lower() or "gateway" in result.lower()

    def test_returns_string(self):
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                200, text="", json_data=self._RETRY_JSON
            )
            result = retry_failed_webhook.invoke({"log_id": "WH-001"})
        assert isinstance(result, str)

    def test_timeout_kwarg_is_set(self):
        """The POST call must use an explicit timeout."""
        with patch("agents.agent_tools.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                200, text="", json_data=self._RETRY_JSON
            )
            retry_failed_webhook.invoke({"log_id": "WH-001"})
        _, kwargs = mock_post.call_args
        assert "timeout" in kwargs


# ──────────────────────────────────────────────────────────────────────────────
# search_knowledge_base
# ──────────────────────────────────────────────────────────────────────────────


def _make_doc(content: str, source: str = "decline_codes.md") -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source}
    return doc


class TestSearchKnowledgeBase:
    def test_returns_concatenated_content(self):
        docs = [
            _make_doc("Code 93 means Risk Block.", "decline_codes.md"),
            _make_doc("Contact the risk team.", "decline_codes.md"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            result = search_knowledge_base.invoke(
                {"query": "What is decline code 93?"}
            )
        assert "Code 93 means Risk Block." in result
        assert "Contact the risk team." in result

    def test_includes_source_labels(self):
        docs = [_make_doc("Passage about code 51.", "decline_codes.md")]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            result = search_knowledge_base.invoke({"query": "code 51"})
        assert "decline_codes.md" in result

    def test_multiple_sources_separated(self):
        docs = [
            _make_doc("Retry policy details.", "webhook_integration.md"),
            _make_doc("T+1 settlement info.", "payout_schedules.md"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            result = search_knowledge_base.invoke({"query": "settlement"})
        assert "webhook_integration.md" in result
        assert "payout_schedules.md" in result
        # Chunks must be separated (not just concatenated without delimiter)
        assert "---" in result

    def test_empty_retrieval_returns_guidance(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            result = search_knowledge_base.invoke({"query": "something obscure"})
        assert "no relevant" in result.lower() or "not found" in result.lower()

    def test_file_not_found_returns_init_message(self):
        with patch(
            "agents.agent_tools.get_retriever",
            side_effect=FileNotFoundError("chroma_db missing"),
        ):
            result = search_knowledge_base.invoke({"query": "code 93"})
        assert "rag_setup.py" in result or "not been initialized" in result.lower()

    def test_generic_exception_returns_error(self):
        with patch(
            "agents.agent_tools.get_retriever",
            side_effect=RuntimeError("something went wrong"),
        ):
            result = search_knowledge_base.invoke({"query": "code 93"})
        assert "unexpected error" in result.lower() or "error" in result.lower()

    def test_returns_string(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_make_doc("some content")]
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            result = search_knowledge_base.invoke({"query": "test"})
        assert isinstance(result, str)

    def test_query_is_forwarded_to_retriever(self):
        """The exact query string must be passed to retriever.invoke()."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_make_doc("x")]
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            search_knowledge_base.invoke(
                {"query": "Why did my transaction fail with code 93?"}
            )
        mock_retriever.invoke.assert_called_once_with(
            "Why did my transaction fail with code 93?"
        )

    def test_dict_input_missing_query_coerced_to_string(self):
        """When the LLM passes a dict without a 'query' key the tool must not crash."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_make_doc("some KB content")]
        # Simulates the buggy LLM call: search_knowledge_base({'decline_code': '93'})
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            result = search_knowledge_base.invoke({"decline_code": "93_Risk_Block"})
        assert isinstance(result, str)
        # retriever must have received a string (coerced from the dict)
        called_with = mock_retriever.invoke.call_args[0][0]
        assert isinstance(called_with, str)

    def test_none_decline_code_dict_does_not_raise(self):
        """{'decline_code': None} — the exact bug input — must not raise ValidationError."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            # Must not raise; must return a non-crashing string
            result = search_knowledge_base.invoke({"decline_code": None})
        assert isinstance(result, str)

    def test_none_decline_code_dict_falls_back_to_safe_query(self):
        """When all dict values are None the fallback query keeps the retriever happy."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("agents.agent_tools.get_retriever", return_value=mock_retriever):
            search_knowledge_base.invoke({"decline_code": None})
        called_with = mock_retriever.invoke.call_args[0][0]
        # Must be a non-empty string — the safe fallback
        assert isinstance(called_with, str)
        assert len(called_with) > 0


# ──────────────────────────────────────────────────────────────────────────────
# fetch_merchant_diagnostics
# ──────────────────────────────────────────────────────────────────────────────


class TestFetchMerchantDiagnostics:
    _TXN_LIST = [
        {"transaction_id": "TXN-001", "status": "SUCCESS", "decline_code": None},
        {"transaction_id": "TXN-002", "status": "DECLINED", "decline_code": "93_Risk_Block"},
        {"transaction_id": "TXN-003", "status": "DECLINED", "decline_code": "93_Risk_Block"},
    ]
    _WH_LIST = [
        {
            "log_id": "WH-001", "transaction_id": "TXN-001",
            "timestamp": "2026-03-07T10:00:05+00:00",
            "event_type": "payment.success", "http_status": 200,
            "delivery_attempts": 1, "latency_ms": 100,
        },
        {
            "log_id": "WH-002", "transaction_id": "TXN-002",
            "timestamp": "2026-03-07T10:01:05+00:00",
            "event_type": "payment.failed", "http_status": 401,
            "delivery_attempts": 3, "latency_ms": 300,
        },
        {
            "log_id": "WH-003", "transaction_id": "TXN-003",
            "timestamp": "2026-03-07T10:02:05+00:00",
            "event_type": "payment.failed", "http_status": 401,
            "delivery_attempts": 3, "latency_ms": 300,
        },
    ]

    def _make_mocks(self):
        return [
            _mock_response(200, json_data=self._TXN_LIST),
            _mock_response(200, json_data=self._WH_LIST),
        ]

    def test_returns_transaction_summary(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = self._make_mocks()
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_2"}
            )
        assert "SUCCESS" in result
        assert "DECLINED" in result

    def test_reports_decline_code_frequency(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = self._make_mocks()
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_2"}
            )
        assert "93_Risk_Block" in result
        assert "2 occurrences" in result

    def test_reports_webhook_status_frequency(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = self._make_mocks()
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_2"}
            )
        assert "401" in result

    def test_includes_log_ids_for_retry_reference(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = self._make_mocks()
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_2"}
            )
        assert "WH-001" in result or "WH-002" in result or "WH-003" in result

    def test_404_returns_not_found_message(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404, text="Not Found")
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_99"}
            )
        assert "not found" in result.lower()
        assert "merchant_id_99" in result

    def test_unexpected_status_returns_error(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.return_value = _mock_response(503, text="Unavailable")
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_1"}
            )
        assert "503" in result or "unexpected" in result.lower()

    def test_connection_error_returns_guidance(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = req_lib.exceptions.ConnectionError("refused")
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_1"}
            )
        assert "connect" in result.lower() or "gateway" in result.lower()

    def test_timeout_returns_timeout_message(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = req_lib.exceptions.Timeout("timed out")
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_1"}
            )
        assert "API_ERROR" in result or "gateway" in result.lower() or "offline" in result.lower()

    def test_generic_request_exception_returns_error(self):
        import requests as req_lib
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = req_lib.exceptions.RequestException("boom")
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_1"}
            )
        assert "API_ERROR" in result or "error" in result.lower() or "gateway" in result.lower()

    def test_returns_string(self):
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = self._make_mocks()
            result = fetch_merchant_diagnostics.invoke(
                {"merchant_id": "merchant_id_2"}
            )
        assert isinstance(result, str)

    def test_tool_has_correct_name(self):
        assert fetch_merchant_diagnostics.name == "fetch_merchant_diagnostics"

    def test_tool_has_non_empty_description(self):
        assert fetch_merchant_diagnostics.description
        assert len(fetch_merchant_diagnostics.description) > 20

    def test_timeout_kwarg_is_set(self):
        """Both GET calls must use explicit timeouts to avoid hanging."""
        with patch("agents.agent_tools.requests.get") as mock_get:
            mock_get.side_effect = self._make_mocks()
            fetch_merchant_diagnostics.invoke({"merchant_id": "merchant_id_2"})
        for call in mock_get.call_args_list:
            assert "timeout" in call[1]
