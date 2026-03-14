"""
agents/support_agent.py – SupportAgent diagnoses payment failures and
                           merchant queries using an LLM with tool-calling.

The agent is given a set of tools (defined in tools.py) and a system prompt
that instructs it to act as a merchant support specialist.  It uses LangChain's
tool-binding mechanism so the LLM can autonomously decide when to call each
tool to gather context before formulating a response.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.tools import fetch_transaction_status, get_merchant_profile

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert merchant support AI assistant for a 
FinTech payment platform. Your responsibilities include:

1. Diagnosing payment failures using transaction logs and error codes.
2. Answering merchant queries about their account status, KYB/KYC progress,
   and fund settlements.
3. Recommending actionable next steps based on your findings.

Always call the relevant tools to fetch up-to-date information before 
answering. Be concise, professional, and empathetic in your responses.
"""


# ──────────────────────────────────────────────────────────────────────────────
# SupportAgent class
# ──────────────────────────────────────────────────────────────────────────────


class SupportAgent:
    """
    Conversational agent for merchant support.

    Usage::

        agent = SupportAgent()
        reply = agent.run("Why did transaction TXN123 fail?")
        print(reply)
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialise the LLM for reasoning-based support.

        Args:
            model_name: Ollama model to use.  Falls back to the ``LLM_MODEL``
                        environment variable, then to ``deepseek-r1``.
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "deepseek-r1")

        # Instantiate the LLM (local Ollama – no API key required)
        # Uses a reasoning-only model (deepseek-r1 by default).
        # Tool calling is NOT used here; the LLM reasons directly.
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,  # deterministic for support use-cases
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        # Available tools (for reference / future ReAct loop implementation)
        self._tools = [fetch_transaction_status, get_merchant_profile]

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def run(self, user_message: str) -> str:
        """
        Process a single merchant query and return the agent's reply.

        This is a *single-turn* implementation.  For multi-turn conversations
        you would maintain a message history and pass it on each call.

        Args:
            user_message: The natural-language question from the merchant.

        Returns:
            The agent's text response.
        """
        # TODO: implement a full ReAct / LangGraph loop that:
        #   1. Calls the LLM with the current message.
        #   2. If the LLM emits tool calls, execute them and feed results back.
        #   3. Repeat until the LLM returns a plain-text final answer.

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Reasoning-only LLM call (no tool binding – deepseek-r1 does
        # not support native tool calling).
        response = self.llm.invoke(messages)

        # For now, return the content directly.
        # A full agentic loop would handle tool_calls here.
        return response.content or "I was unable to generate a response."
