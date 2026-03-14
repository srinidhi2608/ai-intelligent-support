"""
agents/risk_agent.py – RiskAgent evaluates KYB/KYC documents and assesses
                        merchant risk during the onboarding process.

The agent leverages an LLM to interpret document summaries and combine them
with structured risk signals from the FraudDetector ML model.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.tools import get_merchant_profile

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a KYB/KYC risk assessment AI for a FinTech 
payment platform. Your responsibilities include:

1. Evaluating merchant identity and business documents.
2. Detecting inconsistencies or red flags in onboarding submissions.
3. Assigning a risk tier (LOW / MEDIUM / HIGH) with a written justification.
4. Recommending whether to APPROVE, REQUEST_MORE_INFO, or REJECT the merchant.

Use the available tools to gather merchant profile data before making your
assessment. Always provide a structured, auditable rationale.
"""


# ──────────────────────────────────────────────────────────────────────────────
# RiskAgent class
# ──────────────────────────────────────────────────────────────────────────────


class RiskAgent:
    """
    Agent for KYB/KYC document evaluation and merchant risk assessment.

    Usage::

        agent = RiskAgent()
        assessment = agent.evaluate(merchant_id="MERCH_001")
        print(assessment)
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialise the LLM and bind KYB/KYC-specific tools.

        Args:
            model_name: Ollama model to use.  Falls back to the ``LLM_MODEL``
                        environment variable, then to ``deepseek-r1``.
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "deepseek-r1")

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        self._tools = [get_merchant_profile]
        self.llm_with_tools = self.llm.bind_tools(self._tools)

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, merchant_id: str) -> str:
        """
        Evaluate the risk profile of a merchant during onboarding.

        Args:
            merchant_id: The unique identifier of the merchant to assess.

        Returns:
            A structured risk assessment string from the LLM.
        """
        # TODO: extend with document parsing, OCR output, and ML risk scores
        #   from FraudDetector before passing context to the LLM.

        user_message = (
            f"Please perform a KYB/KYC risk assessment for merchant ID: "
            f"{merchant_id}.  Fetch their profile and provide a risk tier "
            f"(LOW/MEDIUM/HIGH) with a recommendation."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self.llm_with_tools.invoke(messages)
        return response.content or "Unable to complete risk assessment."
