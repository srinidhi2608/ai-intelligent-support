# 🏦 Intelligent Merchant Support & Operations — Agentic AI Platform

> **M.Tech Project** | A production-grade, autonomous AI system for diagnosing
> payment failures, handling webhook remediations, and providing merchant
> intelligence — powered entirely by a **local, privacy-preserving LLM**.

---

## 📋 Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Privacy-First Design](#2-privacy-first-design)
3. [Tech Stack](#3-tech-stack)
4. [Directory Structure](#4-directory-structure)
5. [Prerequisites & Local Configuration](#5-prerequisites--local-configuration)
6. [Step-by-Step Run Instructions](#6-step-by-step-run-instructions)
7. [How to Test the Agent — Hero Prompts](#7-how-to-test-the-agent--hero-prompts)
8. [API Endpoints](#8-api-endpoints)
9. [Running Tests](#9-running-tests)
10. [Generate Synthetic Data](#10-generate-synthetic-data)

---

## 1. Project Overview & Architecture

This platform is a **multi-agent AI system** built to act as an autonomous
Tier-2 support layer for a FinTech payment gateway. Rather than routing every
query to a human operator, the system uses an LLM-powered agent that can:

- 🔍 **Diagnose** payment failures by fetching live transaction telemetry
- 📖 **Explain** cryptic ISO 8583 decline codes by querying a local knowledge base (RAG)
- 🔁 **Remediate** failed webhook deliveries by autonomously issuing retry requests
- 🚨 **Detect** anomalies such as integration failures (mass 401 spikes)

### System Architecture

```
 Merchant / Operator Query
         │
         ▼
 ┌───────────────────────────────────────────────────────┐
 │              Streamlit Chat UI  (app.py)              │
 └───────────────────────┬───────────────────────────────┘
                         │
                         ▼
 ┌───────────────────────────────────────────────────────┐
 │        LangChain Agent Orchestrator                   │
 │        (agents/agent_orchestrator.py)                 │
 │                                                       │
 │  LLM: ChatOllama → llama3.1 (local, via Ollama)       │
 │  Architecture: create_tool_calling_agent + AgentExecutor    │
 │               (langchain_classic.agents)                     │
 │                                                       │
 │  Tools bound to the agent:                            │
 │  ┌─────────────────────────────────────────────────┐  │
 │  │  fetch_transaction_logs   (diagnostic)          │  │
 │  │  retry_failed_webhook     (remediation)         │  │
 │  │  search_knowledge_base    (RAG / explanation)   │  │
 │  └─────────────────────────────────────────────────┘  │
 └───────────────────────┬───────────────────────────────┘
                         │  HTTP calls to localhost:8000
         ┌───────────────┴──────────────┐
         ▼                              ▼
 ┌───────────────────┐       ┌──────────────────────────┐
 │  FastAPI Gateway  │       │  ChromaDB Vector Store   │
 │  (main.py)        │       │  (chroma_db/)            │
 │                   │       │                          │
 │  /api/v1/         │       │  Embeddings:             │
 │  transactions/    │       │  all-MiniLM-L6-v2        │
 │  webhooks/        │       │  (HuggingFace, local)    │
 └───────────────────┘       └──────────────────────────┘
         │
         ▼
 ┌───────────────────┐
 │  Synthetic        │
 │  Telemetry CSVs   │
 │  (data/output/)   │
 │  transactions,    │
 │  webhook_logs,    │
 │  merchants        │
 └───────────────────┘
```

---

## 2. Privacy-First Design 🔒

> **No sensitive data ever leaves the local machine.**

This is a deliberate architectural decision. The entire inference stack —
from LLM reasoning to document embeddings — runs locally:

| Component | Technology | Where it runs |
|---|---|---|
| LLM reasoning & tool calling | `llama3.1` via **Ollama** | Local GPU/CPU |
| Document embeddings (RAG) | `all-MiniLM-L6-v2` via HuggingFace | Local CPU |
| Vector store | **ChromaDB** (file-based) | `chroma_db/` directory |
| Transaction telemetry | Synthetic CSVs | `data/output/` directory |

Because `llama3.1` is a native function-calling model, the agent can invoke
tools (fetch data, retry webhooks, query the knowledge base) without relying
on any cloud inference endpoint. **No OpenAI, Anthropic, or Google API key
is required.**

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| 💬 Chat UI | Streamlit |
| ⚙️ API Gateway | FastAPI + Uvicorn |
| 🤖 Agentic AI | LangChain classic agents (`create_tool_calling_agent` + `AgentExecutor` from `langchain_classic`) |
| 🧠 LLM | Ollama — **Llama 3.1** (local, tool-calling capable) |
| 📚 RAG / Knowledge Base | ChromaDB + HuggingFace `all-MiniLM-L6-v2` |
| 🔬 Machine Learning | Scikit-learn, XGBoost |
| 📊 Data | Pandas, Faker |
| 🧪 Testing | Pytest, HTTPX |

---

## 4. Directory Structure

```
ai-intelligent-support/
├── agents/
│   ├── agent_orchestrator.py   # LangChain tool-calling agent (Llama 3.1)
│   ├── agent_tools.py          # fetch_transaction_logs, retry_failed_webhook, search_knowledge_base
│   ├── support_agent.py        # Conversational support agent
│   ├── risk_agent.py           # KYB/KYC risk evaluation agent
│   └── tools.py                # Lower-level tool helpers
├── api/
│   ├── chat.py                 # POST /chat/message endpoint
│   ├── telemetry.py            # GET /api/v1/transactions & /webhooks endpoints
│   └── webhooks.py             # POST /webhook/payment endpoint
├── data/
│   ├── data_access.py          # DataLoader (reads telemetry CSVs into memory)
│   ├── telemetry_generator.py  # Generates synthetic merchants/transactions/webhooks
│   ├── mock_generator.py       # Quick 5-row preview helper
│   └── DATA_DESCRIPTION.md     # Schema, statistics & injected anomaly descriptions
├── models/
│   └── fraud_model.py          # IsolationForest / XGBoost anomaly detection
├── tests/                      # Pytest suite (218+ tests, fully offline/mocked)
├── app.py                      # Streamlit chat interface (Phase 6 UI)
├── main.py                     # FastAPI application entry point
├── rag_setup.py                # Builds the ChromaDB knowledge base (one-time setup)
├── .env.example                # Environment variable template
└── requirements.txt            # Python dependencies
```

---

## 5. Prerequisites & Local Configuration

### 5.1 — Install Ollama

Ollama is the local model runtime that serves `llama3.1` on your machine.

**Download and install Ollama** from the official site:
👉 **[https://ollama.com/download](https://ollama.com/download)**

Follow the installer for your operating system (macOS, Linux, or Windows).
Once installed, verify that the Ollama daemon is running:

```bash
ollama --version
```

### 5.2 — Pull the Llama 3.1 Model

Pull `llama3.1` — the tool-calling LLM used by the agent:

```bash
ollama pull llama3.1
```

> ⏳ This downloads approximately **4.7 GB** on first run. Subsequent starts
> load the model from cache. Once complete, verify it is available:
>
> ```bash
> ollama list
> ```
>
> You should see `llama3.1` in the output.

### 5.3 — Clone the Repository & Create a Virtual Environment

```bash
git clone https://github.com/srinidhi2608/ai-intelligent-support.git
cd ai-intelligent-support
python -m venv .venv
```

**Activate the virtual environment:**

| Platform | Shell | Command |
|---|---|---|
| macOS / Linux | bash / zsh | `source .venv/bin/activate` |
| Windows | PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows | Command Prompt | `.venv\Scripts\activate.bat` |

> **Windows PowerShell tip:** If you see *"running scripts is disabled"*, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.

### 5.4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs all dependencies including the LangChain–Ollama integration:

```bash
# Already included in requirements.txt, but you can also update it manually:
pip install -U langchain-ollama
```

### 5.5 — Configure Environment Variables

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Open `.env` in any text editor. The key variables are:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `llama3.1` | The local model used by the agent for tool-calling |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `APP_HOST` | `0.0.0.0` | FastAPI bind host |
| `APP_PORT` | `8000` | FastAPI bind port |

> ✅ **No external API keys (OpenAI, Anthropic, etc.) are required.** The
> entire system runs on the local `llama3.1` model served by Ollama.

---

## 6. Step-by-Step Run Instructions

The full system requires **three steps** across **two terminal windows**.
Complete Step 1 once, then keep both terminal processes running simultaneously.

---

### Step 1 — One-Time Setup: Build the Knowledge Base 📚

> Run this **once** before starting the application for the first time.
> You only need to repeat it if you delete the `chroma_db/` or `docs/` directories.

In any terminal:

```bash
python rag_setup.py
```

This script:
1. Generates three FinTech Markdown documents under `docs/` (decline codes, webhook rules, payout schedules)
2. Chunks and embeds them using `all-MiniLM-L6-v2` (fully local, no API key)
3. Persists the Chroma vector store to `chroma_db/`

You should see output similar to:
```
✅ Knowledge base built successfully — 3 documents, N chunks indexed.
```

---

### Step 2 — Terminal 1: Start the FastAPI Mock Gateway 🚀

```bash
python main.py
```

This starts the mock FinTech payment gateway. The agent's tools call this
server at `http://localhost:8000` to fetch transaction details, webhook logs,
and to post retry requests.

Verify it is running by visiting:
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** [http://localhost:8000/](http://localhost:8000/) → `{"status": "ok"}`

> ⚠️ **Keep this terminal running.** The Streamlit agent will fail to call its
> tools if the FastAPI server is not reachable.

---

### Step 3 — Terminal 2: Launch the Streamlit Chat UI 💬

Open a **new** terminal (keep Terminal 1 running) and execute:

```bash
streamlit run app.py
```

Streamlit will open the chat interface automatically in your browser at:
**[http://localhost:8501](http://localhost:8501)**

The first query will trigger Ollama to load `llama3.1` into memory, which may
take 10–30 seconds. Subsequent queries will be faster.

---

## 7. How to Test the Agent — Hero Prompts 🧪

Once both services are running, type any of the following prompts into the
Streamlit chat input. Each test is designed to exercise a distinct capability
of the multi-agent system.

---

### 🧪 Test 1 — The Translation Gap (RAG + API Lookup)

**What it proves:** The agent can fetch live API data *and* cross-reference
the local knowledge base (RAG) to translate a cryptic bank code into plain
English — bridging the gap between raw telemetry and actionable merchant guidance.

> Copy and paste this prompt:

```
Hi, my customer's payment for transaction TXN-00194400 failed. What happened, and what does the error mean?
```

**Expected behaviour:**
1. Agent calls `fetch_transaction_logs("TXN-00194400")` → retrieves the declined transaction with `decline_code: "Code 93"`.
2. Agent calls `search_knowledge_base("What does decline code 93 mean?")` → retrieves the RAG document explaining Code 93 as a *Risk Block* issued by the acquiring bank.
3. Agent synthesises both results into a clear, merchant-friendly explanation with recommended next steps.

---

### 🧪 Test 2 — The Actionability Gap (Tool Execution / Webhook Retry)

**What it proves:** The agent can not only diagnose a server-side failure but
also autonomously execute a remediation action — issuing a live POST request
to retry the webhook delivery without any manual intervention.

> Copy and paste this prompt:

```
Did transaction TXN-00000004 go through? If it did, my server was down, so please resend the webhook notification.
```

**Expected behaviour:**
1. Agent calls `fetch_transaction_logs("TXN-00000004")` → finds a `SUCCESS` transaction with a `500` webhook delivery error.
2. Recognising the 500-level error and the merchant's explicit permission, the agent calls `retry_failed_webhook("<log_id>")` → issues a POST to the gateway retry endpoint.
3. Agent confirms the successful re-delivery with the new HTTP status.

---

### 🧪 Test 3 — Proactive Diagnostics (Multi-Agent Reasoning)

**What it proves:** The agent can correlate multiple data points (a flood of
401 errors across many transactions) to diagnose a systemic integration failure
— reasoning beyond a single transaction to identify a root-cause pattern.

> Copy and paste this prompt:

```
Hi, I am merchant_id_2. None of my orders are syncing today and I haven't changed any code. What is wrong?
```

**Expected behaviour:**
1. Agent queries the telemetry API for recent webhook activity for `merchant_id_2`.
2. Agent identifies a massive spike in `401 Unauthorized` errors — a pattern that indicates the merchant's webhook secret has been rotated or expired on the gateway side.
3. Agent calls `search_knowledge_base("Why is my webhook returning 401?")` to retrieve the authoritative explanation.
4. Agent delivers a precise root-cause diagnosis: the webhook signing secret is mismatched, not a code bug — with clear instructions on how to regenerate it in the merchant dashboard.

---

## 8. API Endpoints

The FastAPI gateway (`python main.py`) exposes the following endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/webhook/payment` | Receive a payment lifecycle event |
| `POST` | `/chat/message` | Send a message to the support agent |
| `GET` | `/api/v1/transactions/{txn_id}` | Fetch a single transaction + webhook log |
| `POST` | `/api/v1/webhooks/{log_id}/retry` | Trigger a webhook retry |

Full interactive documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 9. Running Tests

> **Important:** always run pytest from the **project root** (`ai-intelligent-support/`).
> The `pytest.ini` at the root adds the project root to `sys.path`.

```bash
# Run the entire test suite (218+ tests, all run offline with mocked LLM components)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_agent_orchestrator.py -v

# Run a specific test class or function
python -m pytest tests/test_agent_orchestrator.py::TestSystemPrompt -v
python -m pytest tests/test_agent_orchestrator.py::TestSystemPrompt::test_contains_persona -v
```

All tests mock the LLM and external HTTP calls, so **no Ollama server or
FastAPI gateway is required** to run the test suite.

---

## 10. Generate Synthetic Data

The telemetry dataset (merchants, transactions, webhook logs) is generated
synthetically using [Faker](https://faker.readthedocs.io/). Six realistic
anomalies are deliberately injected to give the agent meaningful patterns to
diagnose.

```bash
# Generate the full multimodal telemetry dataset
python data/telemetry_generator.py
# Output: data/output/merchants.csv, transactions.csv, webhook_logs.csv
```

> ℹ️ The FastAPI server automatically loads these CSVs on startup via the
> `DataLoader` class. If the files are missing, the server starts with empty
> DataFrames and logs a warning.

See **[`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md)** for the
complete schema, scale statistics, and descriptions of all six injected
anomalies.