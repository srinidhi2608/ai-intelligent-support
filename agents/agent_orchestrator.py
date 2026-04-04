"""
agents/agent_orchestrator.py – Tool-Calling Agent Orchestrator for Intelligent
                                Merchant Support (Phase 5).

This module sets up a fully autonomous tool-calling agent using
``create_tool_calling_agent`` and ``AgentExecutor`` from ``langchain.agents``.
The agent is bound to the three production tools defined in
``agents.agent_tools`` and orchestrated by a detailed system prompt that
enforces strict tool-usage policies.

The LLM backend is **Ollama** (local), so no cloud API key is required.
A **tool-capable** model (e.g. ``llama3.1``) is used here because the agent
relies on native tool-calling (function-calling) support.  Reasoning-only
models such as ``deepseek-r1`` do **not** support tool calling and must not
be used here.

Usage (interactive console)::

    python -m agents.agent_orchestrator

The script enters a ``while True`` loop, accepting merchant queries from stdin
and printing the agent's final response to stdout.  Type ``exit`` to quit.

Environment variables
---------------------
* ``LLM_MODEL``       – Tool-capable model name (default: ``llama3.1``).
* ``OLLAMA_BASE_URL`` – Ollama server URL (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import os
import sys

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from agents.agent_tools import merchant_support_tools

# ──────────────────────────────────────────────────────────────────────────────
# System prompt – defines the agent's persona and strict tool-usage policies
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an elite Tier-2 FinTech Support Agent. Your job is to diagnose "
    "merchant payment issues, explain errors clearly, and execute remediations.\n\n"
    "You have access to four tools. You MUST use the provided tools to fetch "
    "data before answering. Do not guess or hallucinate.\n\n"
    "## CRITICAL: Autonomous Multi-Step Execution\n\n"
    "You MUST complete ALL required tool calls in a single turn before giving "
    "any response to the user. NEVER emit partial or intermediate messages "
    "such as 'Let me look that up...', 'I will now check...', or 'Based on "
    "the above, let me...' as standalone replies that pause the investigation. "
    "Execute every required tool call first, then write ONE comprehensive "
    "final answer that synthesises all tool results.\n\n"
    "## Mandatory Tool-Chaining Workflows\n\n"
    "**Workflow A — Specific transaction inquiry "
    "(user asks about a transaction ID):**\n"
    "Step 1: Call `fetch_transaction_logs` to retrieve the transaction and "
    "its associated webhook log.\n"
    "Step 2: Check the ``decline_code`` field in the result. "
    "If ``decline_code`` is ``null``, ``None``, or absent, the transaction was "
    "SUCCESSFUL — there is no error code to look up, so SKIP this step and go "
    "directly to Step 3. "
    "Only if ``decline_code`` is a non-null string (e.g. '93_Risk_Block') MUST "
    "you call `search_knowledge_base` with a plain natural-language query string "
    "(e.g. 'What does decline code 93_Risk_Block mean?') — do NOT reply to the "
    "user yet.\n"
    "Step 3: If the webhook log in the result shows a 5xx HTTP error "
    "(500, 502, 503, 504) AND "
    "the user's message implies permission to retry (e.g. 'please resend', "
    "'resend the webhook', 'my server was down'), call `retry_failed_webhook` "
    "with the ``log_id`` from the webhook log — do NOT ask for additional "
    "confirmation.\n"
    "Step 4: After ALL tool calls are complete, write a single response "
    "synthesising all findings.\n\n"
    "**Workflow B — Merchant-level systemic issue "
    "(user identifies as a merchant_id and reports broad failures):**\n"
    "Step 1: Call `fetch_merchant_diagnostics` with the merchant_id.\n"
    "Step 2: If the diagnostic output shows a repeating error code, "
    "call `search_knowledge_base` to look up its meaning.\n"
    "Step 3: Write a single response synthesising all findings.\n\n"
    "## CRITICAL: Output Format Rules\n\n"
    "You are a customer-facing agent. NEVER output raw JSON, Python "
    "dictionaries, or database record IDs directly to the user (unless the "
    "user specifically asks for a transaction ID or log ID). Always synthesise "
    "the tool observations into a polite, empathetic, and professional natural "
    "language response. Do NOT paste raw JSON or raw dictionary output from any "
    "tool call directly into your answer.\n\n"
    "NEVER output raw tool-call notation in your final response — that means "
    "you must NEVER write text like "
    '`{{"name": "tool_name", "parameters": {{...}}}}` '
    "anywhere in your reply. Tool invocations happen behind the scenes and "
    "must never appear in the message you send to the user.\n\n"
    "## General Tool-Usage Rules\n\n"
    "- If a user asks about a transaction, you MUST call the fetch_transaction_logs tool.\n"
    "- You MUST call retry_failed_webhook when the webhook log shows a "
    "5xx error and the merchant has requested a resend (explicitly or implicitly).\n"
    "- NEVER explain a decline code or webhook error code without first calling "
    "`search_knowledge_base` to retrieve the authoritative definition.\n"
    "- The `query` argument for `search_knowledge_base` MUST always be a plain "
    "natural-language string (e.g. 'What does decline code 93 mean?'). "
    "NEVER pass a Python dict, JSON object, a ``None`` value, or raw field data "
    "as the `query` argument.\n\n"
    "## Response Guidelines\n\n"
    "- Be concise, professional, and empathetic.\n"
    "- Always cite the information source (transaction logs, knowledge base) "
    "when explaining a finding.\n"
    "- If you cannot resolve the issue, escalate by clearly stating the next "
    "steps the merchant should take.\n"
)

# ──────────────────────────────────────────────────────────────────────────────
# Agent initialisation
# ──────────────────────────────────────────────────────────────────────────────


def initialize_agent():
    """
    Create and return a LangChain tool-calling agent executor.

    The function:

    1. Instantiates a ``ChatOllama`` LLM backed by a local Ollama server
       using a **tool-capable** model (configurable via the
       ``LLM_MODEL`` env-var, defaulting to ``llama3.1``).
       Reasoning-only models like ``deepseek-r1`` do **not** support
       tool calling and will raise a 400 error if used here.
    2. Builds a ``ChatPromptTemplate`` with the system prompt, human
       message, and agent scratchpad placeholder.
    3. Creates a tool-calling agent with ``create_tool_calling_agent``.
    4. Wraps everything in ``AgentExecutor`` for verbose, step-by-step
       execution.

    Returns:
        An ``AgentExecutor`` instance ready to invoke.
    """
    # ── Initialise the LLM (local Ollama – no API key required) ──────────
    # Must be a tool-capable model (e.g. llama3.1, qwen2.5, mistral).
    # deepseek-r1 does NOT support tool calling.
    model_name = os.environ.get("LLM_MODEL", "llama3.1")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, temperature=0, base_url=base_url)

    # ── Build the prompt template ─────────────────────────────────────────
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # ── Build the tool-calling agent ──────────────────────────────────────
    agent = create_tool_calling_agent(
        llm, tools=merchant_support_tools, prompt=prompt_template
    )

    # ── Wrap in AgentExecutor for verbose, step-by-step execution ────────
    agent_executor = AgentExecutor(
        agent=agent,
        tools=merchant_support_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=25,
    )

    return agent_executor


# Public alias so callers can use either name.
get_agent = initialize_agent


# ──────────────────────────────────────────────────────────────────────────────
# Interactive console loop
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Intelligent Merchant Support – Tool-Calling Agent Console")
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

        # Invoke the tool-calling agent with the merchant's query
        response = agent.invoke({"input": user_input})

        # Extract the final output from the response
        output = response.get("output", "")
        if output:
            print(f"\n🤖 Agent: {output}\n")
        else:
            print("\n⚠️  The agent did not produce a response.\n")
