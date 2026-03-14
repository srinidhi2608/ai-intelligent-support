"""
agents/agent_orchestrator.py – ReAct Agent Orchestrator for Intelligent
                                Merchant Support (Phase 5).

This module sets up a fully autonomous ReAct (Reasoning and Acting) loop using
``create_agent`` from ``langchain.agents``.  The agent is bound to the
three production tools defined in ``agents.agent_tools`` and orchestrated by a
detailed system prompt that enforces strict tool-usage policies.

The LLM backend is **Ollama** (local), so no cloud API key is required.
A **tool-capable** model (e.g. ``llama3.1``) is used here because the ReAct
agent relies on native tool-calling support.  Reasoning-only models such as
``deepseek-r1`` do **not** support tool calling and must not be used here.

Usage (interactive console)::

    python -m agents.agent_orchestrator

The script enters a ``while True`` loop, accepting merchant queries from stdin
and printing the agent's final response to stdout.  Type ``exit`` to quit.

Environment variables
---------------------
* ``TOOL_LLM_MODEL``  – Tool-capable model name (default: ``llama3.1``).
* ``OLLAMA_BASE_URL`` – Ollama server URL (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import os
import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agents.agent_tools import merchant_support_tools

# ──────────────────────────────────────────────────────────────────────────────
# System prompt – defines the agent's persona and strict tool-usage policies
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an elite Tier-2 FinTech Support Agent. Your job is to diagnose "
    "merchant payment issues, explain errors clearly, and execute remediations.\n\n"
    "## Strict Tool-Usage Rules\n\n"
    "1. **ALWAYS** use `search_knowledge_base` to look up the meaning of a "
    "decline code or webhook error **before** explaining it to the user. "
    "Never guess or hallucinate the meaning of an error code.\n\n"
    "2. If a merchant asks about a **specific transaction**, use "
    "`fetch_transaction_logs` **first** to retrieve its details before "
    "responding.\n\n"
    "3. If a webhook log shows a **500-level HTTP error** (500, 502, 503, 504), "
    "ask the merchant for permission to retry. If permission is granted or "
    "implied in the prompt, use `retry_failed_webhook` to remediate the issue.\n\n"
    "## Response Guidelines\n\n"
    "- Be concise, professional, and empathetic.\n"
    "- Always cite the information source (e.g. transaction logs, knowledge base) "
    "when explaining a finding.\n"
    "- If you cannot resolve the issue, escalate by clearly stating the next steps "
    "the merchant should take.\n"
)

# ──────────────────────────────────────────────────────────────────────────────
# Agent initialisation
# ──────────────────────────────────────────────────────────────────────────────


def initialize_agent():
    """
    Create and return a LangGraph ReAct agent executor.

    The function:

    1. Instantiates a ``ChatOllama`` LLM backed by a local Ollama server
       using a **tool-capable** model (configurable via the
       ``TOOL_LLM_MODEL`` env-var, defaulting to ``llama3.1``).
       Reasoning-only models like ``deepseek-r1`` do **not** support
       tool calling and will raise a 400 error if used here.
    2. Binds the three merchant-support tools to it.
    3. Wraps everything in ``create_agent`` (from ``langchain.agents``)
       with the system prompt via the ``system_prompt`` parameter.

    Returns:
        The compiled LangGraph agent executor (a ``CompiledGraph``).
    """
    # ── Initialise the LLM (local Ollama – no API key required) ──────────
    # Must be a tool-capable model (e.g. llama3.1, qwen2.5, mistral).
    # deepseek-r1 does NOT support tool calling.
    model_name = os.environ.get("TOOL_LLM_MODEL", "llama3.1")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(
        model=model_name,
        temperature=0,  # deterministic for support use-cases
        base_url=base_url,
    )

    # ── Build the LangGraph ReAct agent ──────────────────────────────────
    agent_executor = create_agent(
        llm,
        tools=merchant_support_tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent_executor


# ──────────────────────────────────────────────────────────────────────────────
# Interactive console loop
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Intelligent Merchant Support – ReAct Agent Console")
    print("=" * 60)

    try:
        agent = initialize_agent()
    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nAgent initialised successfully.  Type your query below.\n")

    while True:
        try:
            user_input = input("Merchant Query (type 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D / Ctrl-C gracefully
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Invoke the ReAct agent with the merchant's query
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )

        # Extract the final AI message from the response
        ai_messages = response.get("messages", [])
        if ai_messages:
            final_message = ai_messages[-1]
            print(f"\n🤖 Agent: {final_message.content}\n")
        else:
            print("\n⚠️  The agent did not produce a response.\n")
