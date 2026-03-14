"""
app.py – Streamlit Chat Interface for Intelligent Merchant Support (Phase 6).

This module wraps the existing LangGraph ReAct agent in a clean Streamlit chat
UI.  It provides a conversational interface where merchants can ask questions
about transaction declines, webhook errors, and payout schedules, and receive
autonomous, tool-backed answers from the AI agent.

The LLM backend is **Ollama** (local), so no cloud API key is required.
A **tool-capable** model (e.g. ``llama3.1``) is used because the ReAct agent
relies on native tool-calling support.

Usage::

    streamlit run app.py

Environment variables
---------------------
* ``TOOL_LLM_MODEL``  – Tool-capable model name (default: ``llama3.1``).
* ``OLLAMA_BASE_URL`` – Ollama server URL (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import os

import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.agent_tools import merchant_support_tools
from agents.agent_orchestrator import SYSTEM_PROMPT

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
    Initialise and return a LangGraph ReAct agent executor.

    The agent is backed by ``ChatOllama`` using a **tool-capable** model
    (configurable via the ``TOOL_LLM_MODEL`` env-var, default ``llama3.1``)
    and bound to the three merchant-support tools.

    Returns:
        The compiled LangGraph agent executor (a ``CompiledGraph``).
    """
    # Must be a tool-capable model (e.g. llama3.1, qwen2.5, mistral).
    # deepseek-r1 does NOT support tool calling.
    model_name = os.environ.get("TOOL_LLM_MODEL", "llama3.1")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        base_url=base_url,
    )

    agent_executor = create_agent(
        llm,
        tools=merchant_support_tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent_executor


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
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_prompt)]}
            )

            # Extract the final AI message from the response
            ai_messages = response.get("messages", [])
            if ai_messages:
                final_message = ai_messages[-1]
                assistant_reply = final_message.content
            else:
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
