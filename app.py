"""
app.py – Streamlit Chat Interface for Intelligent Merchant Support (Phase 6).

This module wraps the LangChain tool-calling agent in a clean Streamlit chat
UI.  It provides a conversational interface where merchants can ask questions
about transaction declines, webhook errors, and payout schedules, and receive
autonomous, tool-backed answers from the AI agent.

The LLM backend is **Ollama** (local), so no cloud API key is required.
A **tool-capable** model (e.g. ``llama3.1``) is used because the agent
relies on native tool-calling (function-calling) support.

Usage::

    streamlit run app.py

Environment variables
---------------------
* ``LLM_MODEL``       – Tool-capable model name (default: ``llama3.1``).
* ``OLLAMA_BASE_URL`` – Ollama server URL (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import streamlit as st

from agents.agent_orchestrator import SYSTEM_PROMPT, initialize_agent

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinTech AI Support Agent",
    page_icon="🏦",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# Agent initialisation (cached so it survives Streamlit re-runs)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def get_agent():
    """
    Initialise and return a LangChain tool-calling agent executor.

    Delegates to ``initialize_agent()`` from ``agents.agent_orchestrator``
    and caches the result so it survives Streamlit re-runs.

    Returns:
        An ``AgentExecutor`` instance ready to invoke.
    """
    return initialize_agent()


# ──────────────────────────────────────────────────────────────────────────────
# UI header
# ──────────────────────────────────────────────────────────────────────────────

st.title("🏦 FinTech AI Support Agent")
st.markdown(
    "Welcome to the **Intelligent Merchant Support** console.  "
    "This autonomous agent can **diagnose transaction declines**, "
    "**explain webhook errors**, and **retry failed webhook deliveries** — "
    "all backed by real telemetry data and an internal knowledge base.\n\n"
    "Type your question below to get started."
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state – chat history
# ──────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ──────────────────────────────────────────────────────────────────────────────
# User input & agent invocation
# ──────────────────────────────────────────────────────────────────────────────

user_prompt = st.chat_input(
    "Ask a question (e.g., 'Why did transaction txn_123 fail?')"
)

if user_prompt:
    # ── Display and record the user message ──────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ── Invoke the agent with a spinner ──────────────────────────────────
    with st.spinner("Agent is investigating telemetry data..."):
        try:
            agent = get_agent()
            response = agent.invoke({"input": user_prompt})

            # Extract the final output from the response
            assistant_reply = response.get("output", "")
            if not assistant_reply:
                assistant_reply = (
                    "⚠️ The agent did not produce a response. "
                    "Please try rephrasing your question."
                )

        except ValueError as exc:
            assistant_reply = f"⚠️ Configuration error: {exc}"
        except Exception as exc:
            assistant_reply = (
                f"⚠️ An unexpected error occurred: {exc}\n\n"
                "Please ensure the FastAPI gateway is running "
                "(`uvicorn main:app --reload`) and try again."
            )

    # ── Display and record the assistant response ────────────────────────
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
