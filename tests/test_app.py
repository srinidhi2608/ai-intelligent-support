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

    def test_forbids_ill_check_knowledge_base(self):
        """SYSTEM_PROMPT must explicitly forbid 'I'll check our knowledge base'."""
        from agents.agent_orchestrator import SYSTEM_PROMPT

        assert "I'll check our knowledge base" in SYSTEM_PROMPT

    def test_contains_forbidden_intermediate_phrases_section(self):
        """SYSTEM_PROMPT must contain the FORBIDDEN Intermediate Phrases section."""
        from agents.agent_orchestrator import SYSTEM_PROMPT

        assert "FORBIDDEN" in SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# _is_incomplete_response – incomplete-investigation detector
# ──────────────────────────────────────────────────────────────────────────────


class TestIsIncompleteResponse:
    """Test the _is_incomplete_response detector in isolation."""

    def _fn(self, text: str) -> bool:
        from app import _is_incomplete_response
        return _is_incomplete_response(text)

    # ── Positive cases (should detect as incomplete) ──────────────────────

    def test_detects_ill_check_knowledge_base(self):
        text = (
            "Based on the transaction logs, it appears that the payment for "
            "TXN-00194400 was declined with a decline code of '93_Risk_Block'. "
            "To understand what this error means, I'll check our knowledge base."
        )
        assert self._fn(text) is True

    def test_detects_i_will_check_knowledge_base(self):
        text = "I will check the knowledge base for this decline code."
        assert self._fn(text) is True

    def test_detects_let_me_search(self):
        text = "Let me search the knowledge base for decline code 93."
        assert self._fn(text) is True

    def test_detects_ill_query_the_kb(self):
        text = "I'll query the knowledge base to find the meaning."
        assert self._fn(text) is True

    def test_detects_i_need_to_check(self):
        text = "I need to check our knowledge base for this error code."
        assert self._fn(text) is True

    def test_detects_let_me_consult(self):
        text = "Let me consult the documentation for more details."
        assert self._fn(text) is True

    def test_detects_ill_look_up_the_kb(self):
        text = "I'll look up the knowledge base for this code."
        assert self._fn(text) is True

    def test_detects_im_going_to_check(self):
        text = "I'm going to check our knowledge base now."
        assert self._fn(text) is True

    # ── Negative cases (should NOT detect as incomplete) ──────────────────

    def test_clean_prose_not_flagged(self):
        text = (
            "The transaction TXN-00194400 was declined due to risk block "
            "(Code 93). According to our knowledge base, this means..."
        )
        assert self._fn(text) is False

    def test_complete_answer_not_flagged(self):
        text = (
            "The payment for TXN-00194400 was declined with a decline code "
            "of '93_Risk_Block'. Based on our knowledge base, Code 93 "
            "indicates a risk block by the acquiring bank."
        )
        assert self._fn(text) is False

    def test_empty_string_not_flagged(self):
        assert self._fn("") is False

    def test_normal_sentence_with_check_not_flagged(self):
        """A sentence that uses 'check' without the tool-announcement pattern."""
        text = "Please check your API credentials on the merchant dashboard."
        assert self._fn(text) is False

    def test_past_tense_checked_not_flagged(self):
        """Past-tense 'I checked' is not an announcement of future action."""
        text = "I checked the knowledge base and found that Code 93 means..."
        assert self._fn(text) is False


# ──────────────────────────────────────────────────────────────────────────────
# _extract_decline_code – decline-code extractor
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractDeclineCode:
    """Test the _extract_decline_code helper in isolation."""

    def _fn(self, text: str):
        from app import _extract_decline_code
        return _extract_decline_code(text)

    def test_extracts_93_risk_block_single_quotes(self):
        text = "declined with a decline code of '93_Risk_Block'."
        assert self._fn(text) == "93_Risk_Block"

    def test_extracts_93_risk_block_double_quotes(self):
        text = 'declined with a decline code of "93_Risk_Block".'
        assert self._fn(text) == "93_Risk_Block"

    def test_extracts_93_risk_block_no_quotes(self):
        text = "decline_code: 93_Risk_Block in the transaction."
        assert self._fn(text) == "93_Risk_Block"

    def test_extracts_51_insufficient_funds(self):
        text = "The error code is '51_Insufficient_Funds'."
        assert self._fn(text) == "51_Insufficient_Funds"

    def test_extracts_from_full_incomplete_response(self):
        text = (
            "Based on the transaction logs, it appears that the payment for "
            "TXN-00194400 was declined with a decline code of '93_Risk_Block'. "
            "To understand what this error means, I'll check our knowledge base."
        )
        assert self._fn(text) == "93_Risk_Block"

    def test_returns_none_for_no_decline_code(self):
        text = "The transaction was successful with no errors."
        assert self._fn(text) is None

    def test_returns_none_for_empty_string(self):
        assert self._fn("") is None

    def test_returns_none_for_none_input(self):
        # noinspection PyTypeChecker
        assert self._fn(None) is None


# ──────────────────────────────────────────────────────────────────────────────
# _auto_repair_incomplete_response – auto-repair logic
# ──────────────────────────────────────────────────────────────────────────────


class TestAutoRepairIncompleteResponse:
    """Test the _auto_repair_incomplete_response repair strategy."""

    def _fn(self, incomplete_text, user_prompt, agent):
        from app import _auto_repair_incomplete_response
        return _auto_repair_incomplete_response(incomplete_text, user_prompt, agent)

    def test_stage1_direct_kb_lookup_when_decline_code_present(self):
        """When a decline code is found, the function should call
        search_knowledge_base directly and synthesise a complete response."""
        incomplete = (
            "Based on the transaction logs, the payment for TXN-00194400 "
            "was declined with a decline code of '93_Risk_Block'. "
            "To understand what this error means, I'll check our knowledge base."
        )
        mock_agent = MagicMock()
        kb_answer = (
            "[Source 1: decline_codes.md]\n"
            "Code 93 — Risk Block: The acquiring bank has blocked this "
            "transaction due to suspected fraud or risk-management rules."
        )

        with patch(
            "agents.agent_tools.search_knowledge_base"
        ) as mock_kb:
            mock_kb.invoke.return_value = kb_answer
            result = self._fn(incomplete, "What happened to TXN-00194400?", mock_agent)

        # Must contain KB findings
        assert "93_Risk_Block" in result
        assert "knowledge base" in result.lower()
        # Must NOT contain the incomplete phrase
        assert "I'll check our knowledge base" not in result
        # Agent should NOT have been retried (deterministic repair succeeded)
        mock_agent.invoke.assert_not_called()

    def test_stage2_retry_when_no_decline_code(self):
        """When no decline code can be extracted, the function should
        retry the agent with reinforced instructions."""
        incomplete = (
            "There seems to be an issue. I'll check our knowledge base "
            "for more information."
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "The issue has been fully investigated. No errors found."
        }

        result = self._fn(incomplete, "What is wrong?", mock_agent)

        # The retry must have been called
        mock_agent.invoke.assert_called_once()
        # The retry prompt must contain reinforcement language
        call_args = mock_agent.invoke.call_args[0][0]
        assert "CRITICAL INSTRUCTION" in call_args["input"]
        # Result should be the retry's answer
        assert "fully investigated" in result

    def test_returns_original_when_both_stages_fail(self):
        """If both KB lookup and retry fail, return the original text."""
        incomplete = (
            "Something went wrong. I'll check our knowledge base."
        )
        mock_agent = MagicMock()
        # Retry also fails
        mock_agent.invoke.return_value = {
            "output": "I'll check our knowledge base again."
        }

        result = self._fn(incomplete, "Help me", mock_agent)

        # Should return the original since everything failed
        assert result == incomplete

    def test_kb_exception_falls_back_to_retry(self):
        """If KB lookup throws an exception, fall back to retry."""
        incomplete = (
            "The payment was declined with decline code '93_Risk_Block'. "
            "I'll check our knowledge base."
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "Code 93 means risk block. Here is the full explanation."
        }

        with patch(
            "agents.agent_tools.search_knowledge_base"
        ) as mock_kb:
            mock_kb.invoke.side_effect = Exception("KB unavailable")
            result = self._fn(incomplete, "What happened?", mock_agent)

        # Should have fallen back to retry
        mock_agent.invoke.assert_called_once()
        assert "Code 93 means risk block" in result
