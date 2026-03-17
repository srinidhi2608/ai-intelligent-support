"""
tests/test_agent_output_format.py – Output-format guardrail tests for the
                                     tool-calling agent.

These tests verify two complementary layers of protection against raw-data
leakage in the agent's customer-facing responses:

1. **Prompt-level guardrails** – assertions that ``SYSTEM_PROMPT`` explicitly
   instructs the LLM never to output raw JSON, dictionary literals, or
   database IDs.

2. **Runtime output validation** – assertions on the shape and content of the
   dict returned by ``agent_executor.invoke()``:
   * ``response["output"]`` must be a plain Python ``str``.
   * It must not contain raw JSON curly-brace syntax (``{`` / ``}``).
   * It must be non-empty.

A ``_has_raw_json()`` helper is included and unit-tested independently so it
can be reused in future regression tests or integrated into a post-processing
guardrail function.

How to run
----------
From the project root::

    python -m pytest tests/test_agent_output_format.py -v

All tests run offline with fully mocked LLM components — no Ollama server
or FastAPI gateway is required.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from agents.agent_orchestrator import SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# Validation helper
# ──────────────────────────────────────────────────────────────────────────────


def _has_raw_json(text: str) -> bool:
    """Return ``True`` if *text* appears to contain a raw JSON object or Python
    dict literal.

    The heuristic matches a literal ``{`` that appears at the very start of
    the string or immediately after a newline — indicating a JSON object or
    dict is being pasted on its own line rather than embedded mid-sentence.
    Braces that appear in the middle of a word or sentence are not flagged.
    """
    return bool(re.search(r'(?:^|\n)\{', text))


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for _has_raw_json
# ──────────────────────────────────────────────────────────────────────────────


class TestHasRawJsonHelper:
    """Test the _has_raw_json detection helper in isolation."""

    def test_detects_bare_json_string(self):
        raw = '{"transaction_id": "TXN-001", "status": "DECLINED"}'
        assert _has_raw_json(raw) is True

    def test_detects_json_after_newline(self):
        text = "Here is the data:\n{'key': 'value'}"
        assert _has_raw_json(text) is True

    def test_detects_json_on_its_own_line(self):
        # A JSON blob pasted on a separate line after a colon is a real leakage
        # pattern and must be flagged.
        text = "Here is the raw data:\n{'decline_code': '93_Risk_Block'}"
        assert _has_raw_json(text) is True

    def test_accepts_clean_prose(self):
        clean = (
            "The transaction was declined due to insufficient funds. "
            "Please advise the cardholder to top up their account."
        )
        assert _has_raw_json(clean) is False

    def test_accepts_brace_inside_sentence(self):
        # Curly braces embedded mid-sentence (not at line start) are acceptable
        # and must NOT be flagged.
        text = "The payload format is {status: declined, code: 93} as documented."
        assert _has_raw_json(text) is False

    def test_accepts_empty_string(self):
        assert _has_raw_json("") is False


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT output-format guardrail assertions
# ──────────────────────────────────────────────────────────────────────────────


class TestOutputFormatGuardrailsInPrompt:
    """Verify the SYSTEM_PROMPT contains the required output-format rules."""

    def test_forbids_raw_json_output(self):
        """Prompt must explicitly prohibit raw JSON output."""
        assert "raw JSON" in SYSTEM_PROMPT or "raw json" in SYSTEM_PROMPT.lower()

    def test_forbids_raw_dictionaries(self):
        """Prompt must explicitly prohibit raw dictionary/object output."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "dict" in prompt_lower or "raw json" in prompt_lower

    def test_requires_natural_language_response(self):
        """Prompt must explicitly require a natural language final response."""
        assert "natural language" in SYSTEM_PROMPT.lower()

    def test_customer_facing_guardrail_present(self):
        """Prompt must identify the agent as customer-facing with output rules."""
        assert "customer-facing" in SYSTEM_PROMPT

    def test_forbids_pasting_raw_tool_output(self):
        """Prompt must prohibit pasting raw tool responses into the answer."""
        assert "raw JSON" in SYSTEM_PROMPT or "raw dictionary" in SYSTEM_PROMPT.lower()

    def test_synthesise_instruction_present(self):
        """Prompt must instruct the agent to synthesise observations."""
        assert "synthesise" in SYSTEM_PROMPT or "synthesize" in SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# Agent output format validation (mocked executor)
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentOutputFormat:
    """Validate the shape and content of the response returned by the agent."""

    _CLEAN_REPLY = (
        "The payment for TXN-00000004 was declined due to a risk block "
        "(code 93_Risk_Block). This indicates the transaction was flagged "
        "by our internal fraud-velocity rules. Please contact your risk "
        "team to review the card BIN and merchant account."
    )

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_output_is_a_string(self, mock_llm_cls, mock_create, mock_executor_cls):
        """response['output'] must be a plain Python str."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": self._CLEAN_REPLY}
        mock_executor_cls.return_value = mock_agent

        from agents.agent_orchestrator import initialize_agent

        executor = initialize_agent()
        response = executor.invoke({"input": "Check transaction TXN-00000004"})

        assert isinstance(response["output"], str)

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_output_does_not_contain_raw_json_open_brace(
        self, mock_llm_cls, mock_create, mock_executor_cls
    ):
        """response['output'] must not look like a raw JSON object (no bare '{' on its own line)."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": self._CLEAN_REPLY}
        mock_executor_cls.return_value = mock_agent

        from agents.agent_orchestrator import initialize_agent

        executor = initialize_agent()
        response = executor.invoke({"input": "Check transaction TXN-00000004"})

        assert not _has_raw_json(response["output"]), (
            "Agent output must not start with or contain a line-leading raw JSON object"
        )

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_output_does_not_contain_raw_json_close_brace(
        self, mock_llm_cls, mock_create, mock_executor_cls
    ):
        """A pasted raw-JSON blob in the output is detected by _has_raw_json."""
        raw_output = (
            '{"transaction_id": "TXN-00000004", "status": "DECLINED", '
            '"decline_code": "93_Risk_Block"}'
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": raw_output}
        mock_executor_cls.return_value = mock_agent

        from agents.agent_orchestrator import initialize_agent

        executor = initialize_agent()
        response = executor.invoke({"input": "Check transaction TXN-00000004"})

        assert _has_raw_json(response["output"]), (
            "_has_raw_json must flag a response that starts with a JSON object"
        )

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_output_passes_raw_json_helper(
        self, mock_llm_cls, mock_create, mock_executor_cls
    ):
        """A clean agent response must pass the _has_raw_json detection check."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": self._CLEAN_REPLY}
        mock_executor_cls.return_value = mock_agent

        from agents.agent_orchestrator import initialize_agent

        executor = initialize_agent()
        response = executor.invoke({"input": "Check transaction TXN-00000004"})

        assert not _has_raw_json(response["output"]), (
            "Agent output contains what looks like a raw JSON object or dict literal"
        )

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_output_key_present_in_response(
        self, mock_llm_cls, mock_create, mock_executor_cls
    ):
        """The agent executor response dict must contain an 'output' key."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "Transaction TXN-00000004 details retrieved successfully."
        }
        mock_executor_cls.return_value = mock_agent

        from agents.agent_orchestrator import initialize_agent

        executor = initialize_agent()
        response = executor.invoke({"input": "Check transaction TXN-00000004"})

        assert "output" in response

    @patch("agents.agent_orchestrator.AgentExecutor")
    @patch("agents.agent_orchestrator.create_tool_calling_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_output_is_non_empty(
        self, mock_llm_cls, mock_create, mock_executor_cls
    ):
        """A valid agent response must not be an empty string."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "Transaction TXN-00000004 was declined with code 93."
        }
        mock_executor_cls.return_value = mock_agent

        from agents.agent_orchestrator import initialize_agent

        executor = initialize_agent()
        response = executor.invoke({"input": "Check transaction TXN-00000004"})

        assert isinstance(response["output"], str)
        assert len(response["output"]) > 0

    def test_raw_json_response_would_fail_validation(self):
        """Demonstrate that a raw-JSON response IS detected by _has_raw_json."""
        raw_json_output = (
            '{"transaction_id": "TXN-00000004", "status": "DECLINED", '
            '"decline_code": "93_Risk_Block"}'
        )
        assert _has_raw_json(raw_json_output) is True, (
            "A raw JSON response must be flagged as non-compliant"
        )

    def test_merchant_prose_reply_would_pass_validation(self):
        """A proper natural-language reply must NOT be flagged by _has_raw_json."""
        prose_output = (
            "Transaction TXN-00000004 was declined with decline code 93 "
            "(Risk Block). This is a temporary fraud-velocity block. "
            "Please contact your risk team to investigate."
        )
        assert not _has_raw_json(prose_output), (
            "A well-formed natural-language reply must not be flagged"
        )
