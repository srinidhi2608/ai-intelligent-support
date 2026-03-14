"""
agents/agent_orchestrator.py – ReAct Agent Orchestrator for Intelligent
                                Merchant Support (Phase 5).

This module sets up a fully autonomous ReAct (Reasoning and Acting) loop using
LangGraph's ``create_react_agent``.  The agent is bound to the three production
tools defined in ``agents.agent_tools`` and orchestrated by a detailed system
prompt that enforces strict tool-usage policies.

Usage (interactive console)::

    python -m agents.agent_orchestrator

The script enters a ``while True`` loop, accepting merchant queries from stdin
and printing the agent's final response to stdout.  Type ``exit`` to quit.

Environment variables
---------------------
* ``OPENAI_API_KEY`` – Required for the OpenAI backend.
* ``LLM_MODEL``      – Override the model name (default: ``gpt-4o-mini``).
"""

from __future__ import annotations

import os
import sys

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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

    1. Instantiates a ``ChatOpenAI`` LLM (model configurable via the
       ``LLM_MODEL`` env-var, defaulting to ``gpt-4o-mini``).
    2. Binds the three merchant-support tools to it.
    3. Wraps everything in a LangGraph ``create_react_agent`` with the system
       prompt via the ``prompt`` parameter.

    Returns:
        The compiled LangGraph agent executor (a ``CompiledGraph``).

    Raises:
        ValueError: If the ``OPENAI_API_KEY`` environment variable is not set.
    """
    # ── Validate API key early to give a helpful error message ────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set.  "
            "Please export it in your shell or add it to your .env file:\n"
            "  export OPENAI_API_KEY='sk-...'"
        )

    # ── Initialise the LLM ───────────────────────────────────────────────
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,  # deterministic for support use-cases
    )

    # ── Build the LangGraph ReAct agent ──────────────────────────────────
    agent_executor = create_react_agent(
        llm,
        tools=merchant_support_tools,
        prompt=SYSTEM_PROMPT,
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
