"""
tests/test_agent_orchestrator.py – Unit tests for agents/agent_orchestrator.py.

Strategy
--------
* The LLM and LangGraph components are fully mocked so the suite runs offline
  without any API keys.
* We test:
  - The system prompt content (persona, tool-usage rules).
  - ``initialize_agent()`` happy path (returns a compiled agent).
  - ``initialize_agent()`` error path (missing API key raises ``ValueError``).
  - The interactive console loop behaviour (exit, empty input, normal query).
  - The module-level imports and exports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.agent_orchestrator import SYSTEM_PROMPT, initialize_agent


# ──────────────────────────────────────────────────────────────────────────────
# System prompt validation
# ──────────────────────────────────────────────────────────────────────────────


class TestSystemPrompt:
    """Verify the system prompt contains the required persona and rules."""

    def test_contains_persona(self):
        assert "elite Tier-2 FinTech Support Agent" in SYSTEM_PROMPT

    def test_instructs_search_knowledge_base(self):
        assert "search_knowledge_base" in SYSTEM_PROMPT

    def test_instructs_fetch_transaction_logs(self):
        assert "fetch_transaction_logs" in SYSTEM_PROMPT

    def test_instructs_retry_failed_webhook(self):
        assert "retry_failed_webhook" in SYSTEM_PROMPT

    def test_mentions_decline_code_lookup(self):
        assert "decline code" in SYSTEM_PROMPT.lower()

    def test_mentions_500_level_error(self):
        assert "500" in SYSTEM_PROMPT

    def test_prompt_is_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100


# ──────────────────────────────────────────────────────────────────────────────
# initialize_agent()
# ──────────────────────────────────────────────────────────────────────────────


class TestInitializeAgent:
    """Test the ``initialize_agent`` factory function."""

    def test_raises_without_api_key(self):
        """Must raise ValueError when OPENAI_API_KEY is absent."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                initialize_agent()

    def test_error_message_mentions_env_var(self):
        """The error message should guide the user to set the key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="export"):
                initialize_agent()

    @patch("agents.agent_orchestrator.create_react_agent")
    @patch("agents.agent_orchestrator.ChatOpenAI")
    def test_returns_agent_executor(self, mock_llm_cls, mock_create):
        """With a valid key, the function should return the compiled agent."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            agent = initialize_agent()
        assert agent is mock_create.return_value

    @patch("agents.agent_orchestrator.create_react_agent")
    @patch("agents.agent_orchestrator.ChatOpenAI")
    def test_uses_default_model(self, mock_llm_cls, mock_create):
        """Default model should be gpt-4o-mini when LLM_MODEL is not set."""
        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-test-key"}, clear=True
        ):
            initialize_agent()
        mock_llm_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0)

    @patch("agents.agent_orchestrator.create_react_agent")
    @patch("agents.agent_orchestrator.ChatOpenAI")
    def test_respects_llm_model_env(self, mock_llm_cls, mock_create):
        """LLM_MODEL env-var should override the default model name."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-test-key", "LLM_MODEL": "gpt-4o"},
            clear=True,
        ):
            initialize_agent()
        mock_llm_cls.assert_called_once_with(model="gpt-4o", temperature=0)

    @patch("agents.agent_orchestrator.create_react_agent")
    @patch("agents.agent_orchestrator.ChatOpenAI")
    def test_passes_tools_to_react_agent(self, mock_llm_cls, mock_create):
        """All three merchant_support_tools must be passed to create_react_agent."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            initialize_agent()
        _, kwargs = mock_create.call_args
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 3

    @patch("agents.agent_orchestrator.create_react_agent")
    @patch("agents.agent_orchestrator.ChatOpenAI")
    def test_passes_system_prompt_as_prompt(self, mock_llm_cls, mock_create):
        """The system prompt must be forwarded as the prompt parameter."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            initialize_agent()
        _, kwargs = mock_create.call_args
        assert kwargs.get("prompt") == SYSTEM_PROMPT

    @patch("agents.agent_orchestrator.create_react_agent")
    @patch("agents.agent_orchestrator.ChatOpenAI")
    def test_temperature_is_zero(self, mock_llm_cls, mock_create):
        """LLM temperature should be 0 for deterministic support answers."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            initialize_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["temperature"] == 0
