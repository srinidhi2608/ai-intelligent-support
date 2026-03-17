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

import json
import re

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
# Output post-processing – strip raw tool-call JSON leaked by the LLM
# ──────────────────────────────────────────────────────────────────────────────

_AGENT_TOOL_NAMES = frozenset({
    "fetch_transaction_logs",
    "retry_failed_webhook",
    "search_knowledge_base",
    "fetch_merchant_diagnostics",
})


def _strip_tool_call_leakage(text: str) -> str:
    """Remove raw JSON tool-call objects that the LLM leaked into its reply.

    Some LLM backends (e.g. older Ollama builds of llama3.1) emit their
    intended tool invocation as plain text in the final answer instead of
    executing it silently::

        To understand decline code '93_Risk_Block', I'll look it up.
        {"name": "search_knowledge_base", "parameters": {"query": "..."}}

    This function strips every top-level JSON object whose ``"name"`` key
    matches a known agent tool name, collapses any resulting blank lines, and
    returns the sanitised text.  If nothing substantive remains (the entire
    response was a tool-call blob), an empty string is returned and the caller
    should substitute a graceful fallback message.

    Args:
        text: The raw ``output`` string returned by ``AgentExecutor.invoke()``.

    Returns:
        The sanitised text with all tool-call JSON blobs removed.
    """
    if not text:
        return text

    result_parts: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "{":
            # Collect plain text up to the next potential JSON object
            j = text.find("{", i)
            if j == -1:
                result_parts.append(text[i:])
                break
            result_parts.append(text[i:j])
            i = j
            continue

        # Walk forward to find the matching closing brace, respecting strings
        # and nested objects so we handle ``{"parameters": {"query": "..."}}``
        # correctly even when the query text contains punctuation.
        depth = 0
        in_string = False
        escape_next = False
        found_end: int | None = None

        for j in range(i, n):
            c = text[j]
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    found_end = j
                    break

        if found_end is None:
            # No balanced closing brace — treat the rest as plain text
            result_parts.append(text[i:])
            break

        candidate = text[i : found_end + 1]
        is_tool_call = False
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj.get("name") in _AGENT_TOOL_NAMES:
                is_tool_call = True
        except (json.JSONDecodeError, ValueError):
            pass

        if not is_tool_call:
            result_parts.append(candidate)
        # else: silently drop the tool-call blob

        i = found_end + 1

    cleaned = "".join(result_parts)
    # Collapse three-or-more consecutive blank lines into one
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
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
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # ── Invoke the agent, hiding intermediate tool calls in a collapsible
    #    status widget so only the final answer reaches the main chat flow ──
    with st.status("🔍 Agent is investigating...", expanded=False) as status:
        try:
            agent = get_agent()
            response = agent.invoke({"input": user_prompt})

            # Extract the final output from the response
            assistant_reply = response.get("output", "")

            # Strip any raw tool-call JSON the LLM may have leaked into its
            # final answer (e.g. {"name": "search_knowledge_base", ...}).
            assistant_reply = _strip_tool_call_leakage(assistant_reply)

            if not assistant_reply:
                assistant_reply = (
                    "⚠️ The agent did not produce a response. "
                    "Please try rephrasing your question."
                )

            status.update(
                label="✅ Investigation complete",
                state="complete",
                expanded=False,
            )

        except ValueError as exc:
            assistant_reply = f"⚠️ Configuration error: {exc}"
            status.update(label="⚠️ Configuration error", state="error")
        except Exception as exc:
            assistant_reply = (
                f"⚠️ An unexpected error occurred: {exc}\n\n"
                "Please ensure the FastAPI gateway is running "
                "(`uvicorn main:app --reload`) and try again."
            )
            status.update(label="⚠️ An error occurred", state="error")

    # ── Display and record the assistant response ────────────────────────
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(assistant_reply)
