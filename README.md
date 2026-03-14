# Intelligent Merchant Support & Operations using Agentic AI and Machine Learning

An M.Tech project that acts as an autonomous layer to **diagnose payment failures**, handle **merchant onboarding (KYB/KYC)**, and dynamically tune **fraud risk rules** — powered by Agentic AI (LangChain/LangGraph) and Machine Learning (Scikit-learn / XGBoost).

---

## Directory Structure

```
ai-intelligent-support/
├── api/                  # FastAPI routes
│   ├── __init__.py
│   ├── webhooks.py       # Mock endpoints for payment events
│   └── chat.py           # Endpoint for the merchant support chat
├── agents/               # LLM Agent definitions
│   ├── __init__.py
│   ├── support_agent.py  # Diagnoses logs and webhooks
│   ├── risk_agent.py     # Evaluates KYB/KYC documents
│   └── tools.py          # Python functions the agents can call
├── models/               # ML Models
│   ├── __init__.py
│   └── fraud_model.py    # Anomaly detection / XGBoost wrapper
├── data/                 # Data pipelines
│   ├── __init__.py
│   └── mock_generator.py # Script using Faker to generate synthetic transaction logs
├── tests/                # Pytest test suite
│   ├── __init__.py
│   ├── test_webhooks.py
│   ├── test_chat.py
│   ├── test_mock_generator.py
│   └── test_fraud_model.py
├── .env.example          # Environment variables template
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── main.py               # Application entry point
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Agentic AI | LangChain, LangGraph |
| LLM | Ollama (DeepSeek R1, local) |
| Machine Learning | Scikit-learn, XGBoost |
| Data | Pandas, Faker |
| Testing | Pytest, HTTPX |

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/srinidhi2608/ai-intelligent-support.git
cd ai-intelligent-support
python -m venv .venv
```

**Activate the virtual environment — choose the command for your OS / shell:**

| Platform | Shell | Command |
|----------|-------|---------|
| macOS / Linux | bash / zsh | `source .venv/bin/activate` |
| Windows | PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows | Command Prompt (cmd) | `.venv\Scripts\activate.bat` |

> **Windows PowerShell tip:** If you see an error like *"running scripts is disabled on this system"*, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> once in PowerShell and then retry.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Windows Command Prompt:**
```cmd
copy .env.example .env
```

Open `.env` in any text editor and adjust `LLM_MODEL` or `OLLAMA_BASE_URL` if needed
(defaults: `deepseek-r1` on `http://localhost:11434`).

> **Ollama prerequisite:** Make sure you have [Ollama](https://ollama.com/) installed
> and the DeepSeek R1 model pulled:
> ```bash
> ollama pull deepseek-r1
> ```

### 4. Run the API server

```bash
python main.py
# or
uvicorn main:app --reload
```

The interactive API docs will be available at **http://localhost:8000/docs**.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/webhook/payment` | Receive a payment lifecycle event |
| `POST` | `/chat/message` | Send a message to the merchant support agent |

---

## Running Tests

> **Important:** always run pytest from the **project root directory**
> (`ai-intelligent-support/`), not from inside the `tests/` folder.
> The `pytest.ini` at the root adds the project root to `sys.path` so that
> the `models`, `data`, `api`, and `agents` packages are importable.

```bash
# From the project root — run the entire test suite
pytest tests/ -v

# Run a single test file
pytest tests/test_ml_watcher.py -v

# Run a specific test class or function
pytest tests/test_ml_watcher.py::TestEngineerFeatures -v
pytest tests/test_ml_watcher.py::TestEngineerFeatures::test_output_columns -v
```

If you still see `ModuleNotFoundError: No module named 'models'`, make sure
you are in the project root and your virtual environment is activated.

---

## Generate Synthetic Data

```bash
# Full multimodal telemetry dataset (merchants + transactions + webhook logs)
python data/telemetry_generator.py
# Output: data/output/merchants.csv, transactions.csv, webhook_logs.csv

# Quick 5-row preview (unit-testing helper)
python data/mock_generator.py
```

See **[`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md)** for a complete, presentation-ready description of the generated datasets, their schemas, scale statistics, and all six injected anomalies.

---

## Architecture Overview

```
Merchant Query / Payment Event
        │
        ▼
   FastAPI Layer
   (api/webhooks.py, api/chat.py)
        │
        ▼
   Agentic AI Layer
   ┌──────────────────────────────────┐
   │  SupportAgent  │  RiskAgent     │
   │  (LangChain)   │  (LangChain)   │
   └────────┬───────┴────────┬───────┘
            │  calls tools   │
            ▼                ▼
        agents/tools.py  (fetch_transaction_status, get_merchant_profile)
            │
            ▼
   ML Layer (models/fraud_model.py)
   IsolationForest / XGBoostClassifier
```

---

## Roadmap

- [ ] Wire `SupportAgent` into the `/chat/message` endpoint
- [ ] Build a full ReAct loop with LangGraph for multi-step reasoning
- [ ] Add a real database (PostgreSQL / MongoDB) for transaction persistence
- [ ] Implement OCR-based document parsing for KYB/KYC in `RiskAgent`
- [ ] Train the `FraudDetector` on a real labelled dataset
- [ ] Add streaming responses for the chat endpoint
- [ ] Containerise with Docker and add CI/CD pipeline