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
| LLM | OpenAI GPT-4o (or Gemini) |
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

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

```bash
pytest tests/ -v
```

---

## Generate Synthetic Data

```bash
python data/mock_generator.py
```

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