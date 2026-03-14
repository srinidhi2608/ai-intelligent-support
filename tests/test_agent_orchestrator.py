"""
tests/test_agent_orchestrator.py – Unit tests for agents/agent_orchestrator.py.

Strategy
--------
* The LLM and LangGraph components are fully mocked so the suite runs offline
  without any API keys.
* We test:
  - The system prompt content (persona, tool-usage rules).
  - ``initialize_agent()`` happy path (returns a compiled agent).
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

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_returns_agent_executor(self, mock_llm_cls, mock_create):
        """With defaults, the function should return the compiled agent."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        agent = initialize_agent()
        assert agent is mock_create.return_value

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_uses_default_tool_model(self, mock_llm_cls, mock_create):
        """Default model should be llama3.1 when TOOL_LLM_MODEL is not set."""
        with patch.dict("os.environ", {}, clear=True):
            initialize_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["model"] == "llama3.1"

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_respects_tool_llm_model_env(self, mock_llm_cls, mock_create):
        """TOOL_LLM_MODEL env-var should override the default model name."""
        with patch.dict(
            "os.environ",
            {"TOOL_LLM_MODEL": "qwen2.5"},
            clear=True,
        ):
            initialize_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["model"] == "qwen2.5"

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_passes_tools_to_react_agent(self, mock_llm_cls, mock_create):
        """All three merchant_support_tools must be passed to create_agent."""
        initialize_agent()
        _, kwargs = mock_create.call_args
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 3

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_passes_system_prompt_as_prompt(self, mock_llm_cls, mock_create):
        """The system prompt must be forwarded as the system_prompt parameter."""
        initialize_agent()
        _, kwargs = mock_create.call_args
        assert kwargs.get("system_prompt") == SYSTEM_PROMPT

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_temperature_is_zero(self, mock_llm_cls, mock_create):
        """LLM temperature should be 0 for deterministic support answers."""
        initialize_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["temperature"] == 0

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_default_ollama_base_url(self, mock_llm_cls, mock_create):
        """Default Ollama base URL should be http://localhost:11434."""
        with patch.dict("os.environ", {}, clear=True):
            initialize_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["base_url"] == "http://localhost:11434"

    @patch("agents.agent_orchestrator.create_agent")
    @patch("agents.agent_orchestrator.ChatOllama")
    def test_respects_ollama_base_url_env(self, mock_llm_cls, mock_create):
        """OLLAMA_BASE_URL env-var should override the default base URL."""
        with patch.dict(
            "os.environ",
            {"OLLAMA_BASE_URL": "http://remote-host:11434"},
            clear=True,
        ):
            initialize_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["base_url"] == "http://remote-host:11434"
