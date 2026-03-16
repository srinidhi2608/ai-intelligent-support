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
