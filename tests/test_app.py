"""
tests/test_app.py – Unit tests for app.py (Streamlit Chat UI – Phase 6).

Strategy
--------
* All LLM and LangGraph components are fully mocked so the suite runs offline
  without any API keys.
* We test:
  - ``get_agent()`` happy path (returns a compiled agent).
  - The ``create_agent`` is called with ``system_prompt=`` (not ``state_modifier=``).
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

    @patch("app.create_agent")
    @patch("app.ChatOllama")
    def test_returns_agent_executor(self, mock_llm_cls, mock_create):
        """With defaults, the function should return the compiled agent."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        from app import get_agent

        get_agent.clear()
        agent = get_agent()
        assert agent is mock_create.return_value

    @patch("app.create_agent")
    @patch("app.ChatOllama")
    def test_uses_system_prompt_not_state_modifier(self, mock_llm_cls, mock_create):
        """create_agent must be called with ``system_prompt=``, not ``state_modifier=``."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        from app import get_agent

        get_agent.clear()
        get_agent()

        _, kwargs = mock_create.call_args
        assert "system_prompt" in kwargs, "Expected 'system_prompt' keyword argument"
        assert "state_modifier" not in kwargs, (
            "'state_modifier' is deprecated; use 'system_prompt' instead"
        )

    @patch("app.create_agent")
    @patch("app.ChatOllama")
    def test_passes_merchant_support_tools(self, mock_llm_cls, mock_create):
        """All three merchant_support_tools must be passed to create_agent."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        from app import get_agent

        get_agent.clear()
        get_agent()

        _, kwargs = mock_create.call_args
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 3

    @patch("app.create_agent")
    @patch("app.ChatOllama")
    def test_uses_default_tool_model(self, mock_llm_cls, mock_create):
        """Default model should be llama3.1 when TOOL_LLM_MODEL is not set."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        with patch.dict("os.environ", {}, clear=True):
            from app import get_agent

            get_agent.clear()
            get_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["model"] == "llama3.1"

    @patch("app.create_agent")
    @patch("app.ChatOllama")
    def test_respects_tool_llm_model_env(self, mock_llm_cls, mock_create):
        """TOOL_LLM_MODEL env-var should override the default model name."""
        mock_create.return_value = MagicMock(name="compiled_agent")
        with patch.dict(
            "os.environ",
            {"TOOL_LLM_MODEL": "qwen2.5"},
            clear=True,
        ):
            from app import get_agent

            get_agent.clear()
            get_agent()
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["model"] == "qwen2.5"

    @patch("app.create_agent")
    @patch("app.ChatOllama")
    def test_temperature_is_zero(self, mock_llm_cls, mock_create):
        """LLM temperature should be 0 for deterministic support answers."""
        mock_create.return_value = MagicMock(name="compiled_agent")
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
