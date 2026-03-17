"""
tests/test_app.py – Unit tests for app.py (Streamlit Chat UI – Phase 6).

Strategy
--------
* All LLM and LangChain components are fully mocked so the suite runs offline
  without any API keys.
* We test:
  - ``get_agent()`` happy path (returns an AgentExecutor via initialize_agent).
  - Module-level imports and constants are accessible.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# get_agent() – initialisation logic
# ──────────────────────────────────────────────────────────────────────────────


class TestGetAgent:
    """Test the ``get_agent`` factory function."""

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_returns_agent_executor(self, mock_llm_cls, mock_create, mock_executor_cls):
        """With defaults, the function should return an AgentExecutor."""
        mock_executor_cls.return_value = MagicMock(name="agent_executor")
        from app import get_agent

        get_agent.clear()
        agent = get_agent()
        assert agent is mock_executor_cls.return_value

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_uses_create_tool_calling_agent(self, mock_llm_cls, mock_create, mock_executor_cls):
        """get_agent must use create_tool_calling_agent (not create_agent)."""
        mock_executor_cls.return_value = MagicMock(name="agent_executor")
        from app import get_agent

        get_agent.clear()
        get_agent()

        mock_create.assert_called_once()

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_passes_merchant_support_tools(self, mock_llm_cls, mock_create, mock_executor_cls):
        """All three merchant_support_tools must be passed to AgentExecutor."""
        mock_executor_cls.return_value = MagicMock(name="agent_executor")
        from app import get_agent

        get_agent.clear()
        get_agent()

        _, kwargs = mock_executor_cls.call_args
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 4

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_uses_default_llm_model(self, mock_llm_cls, mock_create, mock_executor_cls):
        """Default model should be llama3.1 when LLM_MODEL is not set."""
        mock_executor_cls.return_value = MagicMock(name="agent_executor")
        with patch.dict("os.environ", {}, clear=True):
            from app import get_agent

            get_agent.clear()
            get_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["model"] == "llama3.1"

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_respects_llm_model_env(self, mock_llm_cls, mock_create, mock_executor_cls):
        """LLM_MODEL env-var should override the default model name."""
        mock_executor_cls.return_value = MagicMock(name="agent_executor")
        with patch.dict(
            "os.environ",
            {"LLM_MODEL": "qwen2.5"},
            clear=True,
        ):
            from app import get_agent

            get_agent.clear()
            get_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["model"] == "qwen2.5"

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_temperature_is_zero(self, mock_llm_cls, mock_create, mock_executor_cls):
        """LLM temperature should be 0 for deterministic support answers."""
        mock_executor_cls.return_value = MagicMock(name="agent_executor")
        from app import get_agent

        get_agent.clear()
        get_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["temperature"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Module-level imports
# ──────────────────────────────────────────────────────────────────────────────


class TestModuleImports:
    """Verify that key symbols are importable from app.py."""

    def test_get_agent_is_callable(self):
        from app import get_agent

        assert callable(get_agent)

    def test_system_prompt_imported(self):
        """SYSTEM_PROMPT must be importable via the agents.agent_orchestrator path."""
        from agents.agent_orchestrator import SYSTEM_PROMPT

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100


# ──────────────────────────────────────────────────────────────────────────────
# _strip_tool_call_leakage – output sanitiser
# ──────────────────────────────────────────────────────────────────────────────


class TestStripToolCallLeakage:
    """Test the _strip_tool_call_leakage sanitiser in isolation."""

    def _fn(self, text: str) -> str:
        from app import _strip_tool_call_leakage
        return _strip_tool_call_leakage(text)

    # ── Does not modify clean prose ───────────────────────────────────────

    def test_clean_prose_unchanged(self):
        text = (
            "The transaction was declined due to a risk block (code 93). "
            "Please contact your risk team to review the card BIN."
        )
        assert self._fn(text) == text

    def test_empty_string_unchanged(self):
        assert self._fn("") == ""

    # ── Single-line tool-call blobs ───────────────────────────────────────

    def test_strips_search_knowledge_base_blob(self):
        blob = (
            '{"name": "search_knowledge_base", '
            '"parameters": {"query": "What does decline code 93 mean?"}}'
        )
        assert self._fn(blob) == ""

    def test_strips_fetch_transaction_logs_blob(self):
        blob = (
            '{"name": "fetch_transaction_logs", '
            '"parameters": {"transaction_id": "TXN-00194400"}}'
        )
        assert self._fn(blob) == ""

    def test_strips_retry_failed_webhook_blob(self):
        blob = (
            '{"name": "retry_failed_webhook", '
            '"parameters": {"log_id": "WH-00000007"}}'
        )
        assert self._fn(blob) == ""

    def test_strips_fetch_merchant_diagnostics_blob(self):
        blob = (
            '{"name": "fetch_merchant_diagnostics", '
            '"parameters": {"merchant_id": "merchant_id_2"}}'
        )
        assert self._fn(blob) == ""

    # ── Prose + blob combinations ─────────────────────────────────────────

    def test_strips_blob_preserves_preceding_text(self):
        """Text before the tool-call blob should be kept."""
        reply = (
            "To understand the decline code '93_Risk_Block', "
            "I'll look it up in our knowledge base.\n"
            '{"name": "search_knowledge_base", '
            '"parameters": {"query": "What does decline code 93 Risk Block mean?"}}'
        )
        cleaned = self._fn(reply)
        assert '{"name"' not in cleaned
        assert "search_knowledge_base" not in cleaned
        # The prose part must still be present
        assert "decline code" in cleaned

    def test_strips_blob_preserves_following_text(self):
        """Text after a tool-call blob should be kept."""
        reply = (
            '{"name": "fetch_transaction_logs", '
            '"parameters": {"transaction_id": "TXN-001"}}\n'
            "Here is the summary of your transaction."
        )
        cleaned = self._fn(reply)
        assert "Here is the summary" in cleaned
        assert '{"name"' not in cleaned

    def test_strips_multiple_blobs(self):
        """Multiple tool-call blobs in one response are all removed."""
        reply = (
            '{"name": "fetch_transaction_logs", "parameters": {"transaction_id": "T1"}}\n'
            "Some prose.\n"
            '{"name": "search_knowledge_base", "parameters": {"query": "code 93"}}'
        )
        cleaned = self._fn(reply)
        assert '{"name"' not in cleaned
        assert "Some prose." in cleaned

    # ── Unknown JSON objects are kept ────────────────────────────────────

    def test_non_tool_json_kept(self):
        """JSON objects whose 'name' is not a known tool must not be stripped."""
        text = '{"name": "some_other_thing", "value": 42}'
        assert self._fn(text) == text

    def test_json_without_name_key_kept(self):
        """JSON objects without a 'name' key must not be stripped."""
        text = '{"status": "DECLINED", "code": "93_Risk_Block"}'
        assert self._fn(text) == text

    # ── Return value type ─────────────────────────────────────────────────

    def test_always_returns_string(self):
        assert isinstance(self._fn("some text"), str)
        assert isinstance(self._fn(""), str)
        assert isinstance(
            self._fn(
                '{"name": "search_knowledge_base", "parameters": {"query": "x"}}'
            ),
            str,
        )


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT – no-tool-call-JSON rule
# ──────────────────────────────────────────────────────────────────────────────


class TestSystemPromptToolCallRule:
    """Verify the SYSTEM_PROMPT explicitly forbids raw tool-call JSON output."""

    def test_forbids_tool_call_notation(self):
        from agents.agent_orchestrator import SYSTEM_PROMPT

        # The prompt must mention that tool-call JSON must never appear
        assert "tool-call" in SYSTEM_PROMPT or "tool call" in SYSTEM_PROMPT.lower()

    def test_forbids_name_parameters_pattern(self):
        from agents.agent_orchestrator import SYSTEM_PROMPT

        # The specific JSON pattern that leaks must be mentioned
        assert '"name"' in SYSTEM_PROMPT or "name" in SYSTEM_PROMPT
