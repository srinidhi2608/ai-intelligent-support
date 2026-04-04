# 🏦 Intelligent Merchant Support & Operations

### An Agentic AI Platform for Autonomous FinTech Diagnostics & Remediation

> **M.Tech Capstone Project** — Department of Computer Science & Engineering
>
> A production-grade, dual-pipeline AI system that combines **reactive
> conversational support** with **proactive machine-learning anomaly detection**
> to autonomously diagnose payment failures, remediate webhook delivery errors,
> and detect systemic merchant integration issues — powered entirely by a
> **local, privacy-preserving LLM** with **zero cloud dependency**.

---

## 📋 Table of Contents

| § | Section | Purpose |
|:-:|---|---|
| 1 | [Executive Summary & Architecture](#1-executive-summary--architecture) | High-level design, dual-pipeline overview |
| 2 | [Privacy-First Design](#2-privacy-first-design) | Why no data ever leaves the machine |
| 3 | [Tech Stack](#3-tech-stack) | Languages, frameworks, and models |
| 4 | [Repository Structure](#4-repository-structure) | Directory layout and key modules |
| 5 | [System Setup & Initialization](#5-system-setup--initialization) | Prerequisites, install, first-run |
| 6 | [Demonstration Guide — Act 1: The Reactive Support UI](#6-demonstration-guide--act-1-the-reactive-support-ui) | Live demo of the conversational agent |
| 7 | [Demonstration Guide — Act 2: The Proactive ML Watcher](#7-demonstration-guide--act-2-the-proactive-ml-watcher-the-grand-finale) | ML pipeline + Agentic Handoff demo |
| 8 | [API Reference](#8-api-reference) | FastAPI endpoint catalogue |
| 9 | [Running the Test Suite](#9-running-the-test-suite) | Pytest instructions |
| 10 | [Synthetic Data Generation](#10-synthetic-data-generation) | Telemetry dataset generators |
| 11 | [Troubleshooting](#11-troubleshooting) | Common errors and fixes |

---

## 1. Executive Summary & Architecture

### 1.1 — Core Innovation

This platform is a **multi-agent AI system** built to act as an autonomous
Tier-2 support layer for a FinTech payment gateway. Rather than routing every
query to a human operator, the system uses an LLM-powered agent that can:

| Capability | Description |
|---|---|
| 🔍 **Diagnose** | Fetch live transaction telemetry and pinpoint payment failures |
| 📖 **Explain** | Query a local RAG knowledge base to translate cryptic ISO 8583 decline codes into plain English |
| 🔁 **Remediate** | Autonomously retry failed webhook deliveries via a live POST request |
| 🚨 **Detect & Escalate** | Proactively detect anomalous merchant behaviour (e.g., mass 401 spikes) using unsupervised ML, then trigger the LangChain agent for autonomous investigation |

### 1.2 — LLM: Local Qwen 2.5 Coder via Ollama

The agent is powered by a **local Qwen 2.5 Coder** model served by
[Ollama](https://ollama.com). This model was selected for its **strict,
reliable JSON tool-calling** capability — it outputs well-formed
function-call payloads that LangChain's `AgentExecutor` can parse and
dispatch without hallucination.

> **Key property:** Because the model runs entirely on the local machine,
> **no merchant transaction data, PII, or API keys ever leave the host**.
> No OpenAI, Anthropic, or Google Cloud dependency exists.

The model name is configured via the `LLM_MODEL` environment variable
(see [§ 5](#5-system-setup--initialization)).

### 1.3 — Dual-Pipeline Architecture

The system operates through two complementary pipelines:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   PIPELINE 1 — REACTIVE FLOW (Human-Initiated)                      │
│                                                                      │
│   Merchant Query                                                     │
│        │                                                             │
│        ▼                                                             │
│   Streamlit Chat UI  (app.py)                                        │
│        │                                                             │
│        ▼                                                             │
│   LangChain Agent Orchestrator  (agents/agent_orchestrator.py)       │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │  LLM: Qwen 2.5 Coder (local, via Ollama)                  │     │
│   │  Architecture: create_tool_calling_agent + AgentExecutor   │     │
│   │                                                            │     │
│   │  Bound Tools:                                              │     │
│   │    • fetch_transaction_logs    (diagnostic)                │     │
│   │    • retry_failed_webhook      (remediation)               │     │
│   │    • search_knowledge_base     (RAG / explanation)         │     │
│   │    • fetch_merchant_diagnostics (systemic analysis)        │     │
│   └────────────────────────────────────────────────────────────┘     │
│        │  HTTP calls                                                 │
│        ▼                                                             │
│   FastAPI Gateway  (main.py @ localhost:8000)                        │
│        │                                                             │
│        ▼                                                             │
│   Telemetry CSVs  (data/output/)  +  ChromaDB Vector Store           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   PIPELINE 2 — PROACTIVE FLOW (Machine-Initiated)                    │
│                                                                      │
│   Telemetry Data  (transactions.csv)                                 │
│        │                                                             │
│        ▼                                                             │
│   Feature Engineering  (rolling-window aggregation)                  │
│        │                                                             │
│        ▼                                                             │
│   Isolation Forest  (unsupervised anomaly detection)                 │
│        │                                                             │
│        ▼  anomaly detected?                                          │
│   Agentic Handoff  →  trigger_agent_for_anomaly()                    │
│        │                                                             │
│        ▼                                                             │
│   LangChain Agent  →  Autonomous Investigation & Resolution         │
│   (same agent as Pipeline 1, invoked programmatically)               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Pipeline 1 (Reactive):** A merchant types a natural-language query into
the Streamlit chat UI. The LangChain agent autonomously chains tool calls
(API lookups, RAG queries, webhook retries) and returns a synthesised answer.

**Pipeline 2 (Proactive):** The `ml_advanced_pipeline.py` script ingests
transaction telemetry, engineers time-series features, and runs an Isolation
Forest to detect anomalous merchant windows. When an anomaly is confirmed,
the script **programmatically invokes the same LangChain agent** via
`trigger_agent_for_anomaly()`, passing a structured system alert. The agent
then diagnoses the issue without any human intervention.

---

## 2. Privacy-First Design 🔒

> **No sensitive data ever leaves the local machine.**

| Component | Technology | Execution Environment |
|---|---|---|
| LLM reasoning & tool calling | **Qwen 2.5 Coder** via Ollama | Local GPU / CPU |
| Document embeddings (RAG) | `all-MiniLM-L6-v2` via HuggingFace | Local CPU |
| Vector store | **ChromaDB** (file-based) | `chroma_db/` directory |
| Transaction telemetry | Synthetic CSVs | `data/output/` directory |

Because the model supports native function/tool calling, the agent can
invoke tools (fetch data, retry webhooks, query the knowledge base) without
relying on any cloud inference endpoint. **No external API keys are required.**

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| 💬 Chat UI | Streamlit |
| ⚙️ API Gateway | FastAPI + Uvicorn |
| 🤖 Agentic AI | LangChain (`create_tool_calling_agent` + `AgentExecutor`) |
| 🧠 LLM | **Qwen 2.5 Coder** — local, via Ollama (tool-calling capable) |
| 📚 RAG / Knowledge Base | ChromaDB + HuggingFace `all-MiniLM-L6-v2` |
| 🔬 Machine Learning | Scikit-learn (Isolation Forest), XGBoost, Pandas, NumPy |
| 📊 Data Generation | Faker (synthetic telemetry with Pareto distributions) |
| 🧪 Testing | Pytest (367+ offline tests), HTTPX |

---

## 4. Repository Structure

```
ai-intelligent-support/
│
├── agents/
│   ├── agent_orchestrator.py      # LangChain tool-calling agent + get_agent()
│   ├── agent_tools.py             # 4 tools: fetch_transaction_logs, retry_failed_webhook,
│   │                              #          search_knowledge_base, fetch_merchant_diagnostics
│   ├── support_agent.py           # Conversational support agent
│   ├── risk_agent.py              # KYB/KYC risk evaluation agent
│   └── tools.py                   # Lower-level tool helpers
│
├── api/
│   ├── chat.py                    # POST /chat/message endpoint
│   ├── telemetry.py               # GET /api/v1/transactions & webhook endpoints
│   └── webhooks.py                # POST /webhook/payment endpoint
│
├── data/
│   ├── data_access.py             # DataLoader (in-memory CSV access)
│   ├── telemetry_generator.py     # Generates synthetic telemetry (~260K rows)
│   ├── mock_generator.py          # Quick 5-row preview helper
│   └── DATA_DESCRIPTION.md        # Schema & injected anomaly descriptions
│
├── models/
│   ├── ml_watcher.py              # Proactive ML Watcher (Isolation Forest)
│   └── fraud_model.py             # IsolationForest / XGBoost anomaly detection
│
├── pages/
│   ├── 1_📊_Analytics.py          # Streamlit analytics dashboard
│   └── 2_🧠_ML_Comparison.py      # ML model comparison dashboard (IF vs. SVM vs. LOF)
│
├── tests/                         # Pytest suite (367+ tests, fully offline / mocked)
│
├── app.py                         # Streamlit chat interface (Reactive UI)
├── ml_advanced_pipeline.py        # ★ Advanced ML Pipeline + Agentic Handoff
├── generate_realistic_data.py     # Realistic data generator (Pareto, MCC, seasonality)
├── main.py                        # FastAPI application entry point
├── rag_setup.py                   # Builds the ChromaDB knowledge base (one-time)
├── .env.example                   # Environment variable template
└── requirements.txt               # Python dependencies
```

---

## 5. System Setup & Initialization

### 5.1 — Prerequisites

| Prerequisite | Installation |
|---|---|
| **Python 3.10+** | [python.org/downloads](https://www.python.org/downloads/) |
| **Ollama** | [ollama.com/download](https://ollama.com/download) |
| **Git** | [git-scm.com](https://git-scm.com/) |

### 5.2 — Clone, Install & Configure

```bash
# 1. Clone the repository
git clone https://github.com/srinidhi2608/ai-intelligent-support.git
cd ai-intelligent-support

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Copy the environment template
cp .env.example .env             # macOS / Linux
# Copy-Item .env.example .env    # Windows PowerShell
```

Open `.env` and set the model:

```dotenv
LLM_MODEL=qwen2.5-coder
OLLAMA_BASE_URL=http://localhost:11434
```

### 5.3 — One-Time Setup: Build the Knowledge Base

```bash
python rag_setup.py
```

This embeds three FinTech Markdown documents (decline codes, webhook rules,
payout schedules) into a local ChromaDB vector store using
`all-MiniLM-L6-v2`. Run this once; repeat only if you delete `chroma_db/`.

### 5.4 — One-Time Setup: Generate Synthetic Telemetry Data

```bash
python generate_realistic_data.py
```

This generates ~50,000 transactions over 7 days with Pareto-distributed
merchant volumes, MCC-based amounts, and time-of-day seasonality. Three
critical demo anomalies are injected automatically (see [§ 10](#10-synthetic-data-generation)).

### Step 1 — Start the API Gateway

```bash
uvicorn main:app --reload --port 8000
```

Verify: [http://localhost:8000/](http://localhost:8000/) → `{"status": "ok"}`

> ⚠️ **Keep this terminal running.** All agent tool calls route through this
> gateway.

### Step 2 — Ensure Ollama Is Running with the Model

In a separate terminal:

```bash
ollama run qwen2.5-coder
```

This loads the model into memory and starts the Ollama inference server.
Once you see the interactive prompt, the model is ready. You can type `/bye`
to exit the interactive session — the model remains loaded in the background
for the agent to use.

> To verify the model is available:
> ```bash
> ollama list
> ```

---

## 6. Demonstration Guide — Act 1: The Reactive Support UI

> **Objective:** Prove that the LLM agent can autonomously diagnose a payment
> failure, explain the error, and execute a webhook retry — all from a single
> natural-language prompt.

### 6.1 — Launch the Streamlit UI

Open a **new terminal** (keep the API gateway running) and execute:

```bash
streamlit run app.py
```

The chat interface opens at **[http://localhost:8501](http://localhost:8501)**.

### 6.2 — The Hero Prompt

Copy and paste the following prompt into the Streamlit chat input:

```
I am merchant_id_5. My server was down. Did TXN-00000004 go through? If yes, please retry the webhook.
```

### 6.3 — What to Observe (Key Points for the Panel)

1. **Click the "🔍 Agent is investigating..." expander** in the Streamlit UI.
   This reveals the agent's full reasoning chain in real time:
   - The agent calls `fetch_transaction_logs("TXN-00000004")` — proving it
     can extract the transaction ID from natural language and fetch live API data.
   - The agent detects that the transaction was `SUCCESS` but the webhook
     delivery returned an `HTTP 500` error.
   - The agent calls `retry_failed_webhook("<log_id>")` — proving the LLM is
     **autonomously executing the `retry_webhook` tool** to remediate the
     failure without human intervention.

2. **The final synthesised response** in the chat bubble:
   - The agent confirms the transaction status, explains the webhook failure,
     and reports the retry result — all in professional, merchant-friendly
     natural language (no raw JSON or internal IDs leaked).

> **What this proves to the panel:** The local Qwen 2.5 Coder model, via
> strict JSON tool-calling, can chain multiple API calls autonomously. The
> agent acts as a fully autonomous Tier-2 support engineer — diagnosing,
> reasoning, and remediating in a single turn.

### 6.4 — Additional Hero Prompts

| # | Prompt | Capability Tested |
|:-:|---|---|
| 1 | `Hi, my customer's payment for transaction TXN-00194400 failed. What happened, and what does the error mean?` | RAG lookup — agent fetches the transaction (decline code `93_Risk_Block`), then queries the knowledge base to explain the code |
| 2 | `Hi, I am merchant_id_2. None of my orders are syncing today and I haven't changed any code. What is wrong?` | Systemic diagnosis — agent calls `fetch_merchant_diagnostics`, identifies the 401 spike, queries the KB, and prescribes a fix |

---

## 7. Demonstration Guide — Act 2: The Proactive ML Watcher (The Grand Finale)

> **Objective:** Prove that the system can **proactively** detect anomalies in
> merchant transaction data using unsupervised machine learning, and then
> **autonomously hand off** the investigation to the LangChain agent — with
> zero human intervention.

### 7.1 — Run the Advanced ML Pipeline

Open a **separate terminal** (keep the API gateway and Ollama running) and
execute:

```bash
python ml_advanced_pipeline.py
```

### 7.2 — What the Script Does (Expected Output for the Panel)

The script executes the following stages, with output printed to the terminal:

#### Stage 1: Automated Edge-Case Tests

The `test_edge_cases()` function constructs a synthetic feature matrix of
20 merchants: 19 healthy merchants with normal transaction volumes and low
decline ratios, and **one anomalous merchant** (`merchant_id_2`) exhibiting a
simulated **401 authentication-failure spike** (150 transactions, 148 declined,
decline ratio ≈ 98.7%, average amount ₹2.50 — a classic card-testing pattern).

The Isolation Forest is trained on this data and the script **asserts** that
`merchant_id_2` is correctly flagged as an anomaly (prediction = 1):

```
✅ Positive Case 1 PASSED: merchant_id_2 (401 webhook spike) correctly flagged as anomaly.
```

> **What this proves:** The Isolation Forest correctly identifies the extreme
> outlier — a merchant with a near-100% decline ratio and micro-amounts —
> validating the model's effectiveness on FinTech-specific anomaly patterns.

#### Stage 2: The Agentic Handoff (The Key Innovation)

Immediately after the assertion passes, the script calls:

```python
trigger_agent_for_anomaly(merchant_2_row)
```

This function:

1. Imports and initialises the **same LangChain agent** used by the Streamlit UI
   (via `from agents.agent_orchestrator import get_agent`).
2. Constructs a structured system alert:
   ```
   SYSTEM ALERT: The ML Watcher has detected a high volume of failed
   transactions for merchant_id_2. Please investigate the root cause
   and take necessary action.
   ```
3. Passes the alert to `agent_executor.invoke({"input": alert_message})`.
4. Prints the agent's final diagnostic response to the terminal.

> **What this proves to the panel:** The ML pipeline and the LangChain agent
> are **fully integrated**. The system detects the anomaly with unsupervised
> ML, then **autonomously triggers the AI agent** to diagnose and prescribe a
> remediation — completing the end-to-end proactive loop without any human
> clicking a button or typing a prompt.

### 7.3 — ML Model Comparison Dashboard (Supplementary)

For a visual comparison of unsupervised models, launch the Streamlit
multi-page app and navigate to the **🧠 ML Comparison** page:

```bash
streamlit run app.py
# → Navigate to "🧠 ML Comparison" in the sidebar
```

This dashboard trains, evaluates, and visually compares three unsupervised
anomaly-detection models on the merged transaction + webhook data:

| Model | Description |
|---|---|
| **Isolation Forest** | Tree-based anomaly isolation (best F1-Score for FinTech anomalies) |
| **One-Class SVM** | Kernel-based novelty detection |
| **Local Outlier Factor** | Density-based local outlier scoring |

Ground truth is derived from known anomaly indicators:
`decline_code == '93_Risk_Block'` **or** `http_status == 401`.

> **Key finding:** Isolation Forest consistently yields the **best F1-Score**
> for FinTech anomaly patterns, which is why it was selected as the production
> model in the proactive pipeline.

---

## 8. API Reference

The FastAPI gateway (`uvicorn main:app`) exposes:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check → `{"status": "ok"}` |
| `POST` | `/webhook/payment` | Receive a payment lifecycle event |
| `POST` | `/chat/message` | Send a message to the support agent |
| `GET` | `/api/v1/transactions/{txn_id}/details` | Fetch transaction + webhook log |
| `POST` | `/api/v1/webhooks/{log_id}/retry` | Trigger a webhook retry |

Full interactive documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 9. Running the Test Suite

```bash
# From the project root (ai-intelligent-support/)
python -m pytest tests/ -v
```

- **367+ tests** across 14 test files
- All tests run **fully offline** with mocked LLM components — no Ollama
  server or FastAPI gateway required
- Covers: agent orchestrator, tool validation, output format guardrails,
  RAG setup, data access, API endpoints, Streamlit UI, ML models

```bash
# Run a specific test file
python -m pytest tests/test_agent_orchestrator.py -v

# Run a specific test class
python -m pytest tests/test_agent_orchestrator.py::TestSystemPrompt -v
```

---

## 10. Synthetic Data Generation

Two generators are available. Both output to `data/output/` with identical
schemas:

### Primary Generator (Recommended)

```bash
python generate_realistic_data.py
```

| Feature | Description |
|---|---|
| **~50K transactions** | 7-day window with realistic volume |
| **Pareto merchant volumes** | Heavy-tail distribution mimicking real gateway traffic |
| **MCC-based amounts** | Category-specific Pareto shape/scale (grocery ₹300–₹2K, electronics ₹2K–₹30K) |
| **Time-of-day seasonality** | Hourly weight curve modelling IST business hours |
| **3 injected demo anomalies** | `TXN-00194400` (93_Risk_Block), `TXN-00000004`/`WH-00000004` (HTTP 500 webhook), 150× `merchant_id_2` auth-failure webhooks |

### Alternative Generator

```bash
python data/telemetry_generator.py
```

Generates ~260,000 rows over a 24-hour window with six injected anomalies.

> The `DataLoader`, FastAPI endpoints, and Streamlit UI work with either
> dataset. See [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md) for
> complete schema documentation.

---

## 11. Troubleshooting

### `ConnectionRefusedError` — API Gateway Not Running

**Symptom:** `API_ERROR: The FinTech Gateway is currently offline or unreachable.`

**Fix:** Start the FastAPI gateway in a separate terminal:

```bash
uvicorn main:app --reload --port 8000
```

### `Input to ChatPromptTemplate is missing variables {"name"}`

**Cause:** Unescaped curly braces in the system prompt. All literal braces in
`SYSTEM_PROMPT` must be doubled (`{{` / `}}`). This is already handled in the
current codebase.

### Knowledge Base Not Initialised

**Symptom:** Agent responds with *"The knowledge base has not been initialized yet."*

**Fix:**

```bash
python rag_setup.py
```

### Model Not Found / Ollama Connection Error

**Fix:** Ensure Ollama is running and the model is pulled:

```bash
ollama run qwen2.5-coder
```