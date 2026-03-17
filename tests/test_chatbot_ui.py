"""
tests/test_chatbot_ui.py – Streamlit UI smoke-tests using AppTest.

Strategy
--------
* Uses Streamlit's native testing framework (``streamlit.testing.v1.AppTest``)
  to load ``app.py`` in a headless runner and exercise the chat UI without a
  running Ollama server or FastAPI gateway.
* The LangChain agent (``initialize_agent``) is fully mocked at import time so
  no network connections are required.
* Key assertions:
  - The app renders without raising an unhandled exception.
  - After a user submits a message, the chat history grows (the response is
    appended to ``st.session_state.messages``).
  - The app handles an offline backend gracefully (no raw exception reaches
    the UI).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_OFFLINE_ERROR_MSG = (
    "API_ERROR: The FinTech Gateway is currently offline or unreachable. "
    "Please inform the merchant that system diagnostics are currently unavailable."
)

_NORMAL_REPLY = (
    "Transaction TXN-00194400 was declined due to a risk block (Code 93). "
    "Please contact your risk team for further investigation."
)


def _make_agent_executor(output: str) -> MagicMock:
    """Build a mock AgentExecutor whose .invoke() returns *output*."""
    executor = MagicMock()
    executor.invoke.return_value = {"output": output}
    return executor


# ──────────────────────────────────────────────────────────────────────────────
# AppTest – offline backend resilience
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentHandlesOfflineBackend:
    """Verify the Streamlit UI stays healthy when the FastAPI gateway is offline."""

    def _run_app_with_input(self, agent_output: str, user_text: str):
        """
        Run ``app.py`` through AppTest with *user_text* entered in the chat box.

        The LangChain ``initialize_agent`` function is monkey-patched so the
        real Ollama / FastAPI calls never fire.  All ``@st.cache_resource``
        caches are cleared globally before each run so every test gets a
        freshly-mocked agent.

        Returns the ``AppTest`` instance after execution for assertion.
        """
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        # Clear ALL st.cache_resource caches — this is required for test
        # isolation.  Without it, @st.cache_resource persists the mock agent
        # from the first test that called get_agent(), causing every subsequent
        # test to reuse that cached mock regardless of the patch applied here.
        # st.cache_resource.clear() resets every cache-resource store in the
        # current process, ensuring each test creates a fresh agent via the
        # patched initialize_agent().
        st.cache_resource.clear()

        mock_executor = _make_agent_executor(agent_output)

        with patch(
            "agents.agent_orchestrator.initialize_agent",
            return_value=mock_executor,
        ):
            at = AppTest.from_file("app.py")
            at.run()
            # Simulate the user typing into the chat input and pressing Enter
            at.chat_input[0].set_value(user_text).run()

        return at

    def test_app_renders_without_exception(self):
        """The Streamlit app must load and render without raising any exception."""
        from streamlit.testing.v1 import AppTest

        mock_executor = _make_agent_executor(_NORMAL_REPLY)
        with patch(
            "agents.agent_orchestrator.initialize_agent",
            return_value=mock_executor,
        ):
            at = AppTest.from_file("app.py")
            at.run()

        assert not at.exception

    def test_chat_history_updates_after_user_input(self):
        """After a user message the session state must contain at least two messages."""
        at = self._run_app_with_input(
            agent_output=_NORMAL_REPLY,
            user_text="Check transaction TXN-00194400",
        )

        assert not at.exception
        messages = at.session_state["messages"] if "messages" in at.session_state else []
        # At minimum: one user message + one assistant reply
        assert len(messages) >= 2

    def test_user_message_recorded_in_history(self):
        """The user's exact input text must appear in session_state.messages."""
        user_text = "Check transaction TXN-00194400"
        at = self._run_app_with_input(
            agent_output=_NORMAL_REPLY,
            user_text=user_text,
        )

        assert not at.exception
        messages = at.session_state["messages"] if "messages" in at.session_state else []
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any(user_text in m.get("content", "") for m in user_messages)

    def test_assistant_reply_recorded_in_history(self):
        """An assistant reply must be appended to session_state.messages."""
        at = self._run_app_with_input(
            agent_output=_NORMAL_REPLY,
            user_text="Check transaction TXN-00194400",
        )

        assert not at.exception
        messages = at.session_state["messages"] if "messages" in at.session_state else []
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_messages) >= 1

    def test_offline_backend_does_not_crash_app(self):
        """When the gateway returns an API_ERROR the UI must not show an exception."""
        at = self._run_app_with_input(
            agent_output=_OFFLINE_ERROR_MSG,
            user_text="Check transaction TXN-00194400",
        )

        assert not at.exception

    def test_offline_backend_message_shown_to_user(self):
        """The API_ERROR message from an offline gateway must reach the chat."""
        at = self._run_app_with_input(
            agent_output=_OFFLINE_ERROR_MSG,
            user_text="Check transaction TXN-00194400",
        )

        assert not at.exception
        messages = at.session_state["messages"] if "messages" in at.session_state else []
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_messages) >= 1
        # The offline error text must appear in the assistant reply
        combined = " ".join(m.get("content", "") for m in assistant_messages)
        assert "API_ERROR" in combined or "offline" in combined.lower() or "unavailable" in combined.lower()

    def test_empty_agent_output_shows_fallback(self):
        """If the agent returns an empty string the UI must show a fallback message."""
        at = self._run_app_with_input(
            agent_output="",
            user_text="Check transaction TXN-00194400",
        )

        assert not at.exception
        messages = at.session_state["messages"] if "messages" in at.session_state else []
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_messages) >= 1
        content = assistant_messages[-1].get("content", "")
        # Must not display a blank bubble — must show the fallback warning
        assert content.strip() != ""
