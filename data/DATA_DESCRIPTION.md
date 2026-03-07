# Synthetic Telemetry Dataset — Presentation Description

> **Project:** Intelligent Merchant Support & Operations using Agentic AI and Machine Learning  
> **M.Tech Capstone | Data Engineering Layer**

---

## 1. Overview

To test and validate the AI agent without exposing real customer data, a **fully synthetic, multi-table payment-gateway telemetry dataset** was generated using Python (Faker + NumPy + Pandas).  
The dataset faithfully replicates the data model of a production payment gateway and includes deliberately injected anomalies that challenge the agent to diagnose six distinct, real-world failure scenarios.

| Attribute | Value |
|-----------|-------|
| Generation script | `data/telemetry_generator.py` |
| Reproducibility seed | `42` (deterministic output) |
| Locale | Indian (`en_IN`) — INR currency, Indian business names |
| Time window | Rolling 24 hours ending at script execution time |
| Total records | **~259,800 transactions + ~259,800 webhook logs** |

---

## 2. Dataset 1 — `merchants.csv`

### Purpose
Master reference table for all payment merchants operating on the gateway.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `merchant_id` | string | Stable unique identifier (`merchant_id_1` … `merchant_id_25`) |
| `business_name` | string | Faker-generated Indian company name |
| `mcc_code` | string (4-digit) | ISO 18245 Merchant Category Code (e.g., `5812` = Restaurants) |
| `webhook_url` | string | HTTPS endpoint that receives real-time payment event notifications |

### Key Statistics
- **25 unique merchants**, each with a distinct webhook endpoint
- MCC codes span 10 categories: Grocery, Restaurants, Retail, Software, Electronics, Telecom, Pharmacy, Hotels, Department Stores, Shoe Stores
- Merchant IDs follow a predictable `merchant_id_<N>` pattern, making it easy to isolate specific merchants during agent testing

### Slide Talking Point
> *"The merchants table acts as the dimension table in our data model, providing the gateway with the profile of every business it processes payments for. Each merchant has a webhook URL — the mechanism through which the gateway pushes real-time event notifications."*

---

## 3. Dataset 2 — `transactions.csv`

### Purpose
The core fact table. Captures every payment attempt processed by the gateway over the last 24 hours.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique ID (`TXN-00000000` sequential, `TXN-CARD-` and `TXN-SPIKE-` for anomaly rows) |
| `merchant_id` | string | Foreign key → `merchants.csv` |
| `timestamp` | datetime (UTC) | When the transaction was initiated |
| `amount` | float | Transaction value in INR (₹100 – ₹50,000 for normal; ₹1 – ₹5 for card-testing anomaly) |
| `currency` | string | Always `INR` |
| `status` | string | `SUCCESS` or `DECLINED` |
| `decline_code` | string / null | Banking ISO decline reason code; `null` if `SUCCESS` |
| `card_bin` | string (6-digit) | Bank Identification Number of the card used |

### Key Statistics

| Metric | Value |
|--------|-------|
| Base transaction rate | **3 transactions per second** |
| Total baseline rows | 3 TPS × 86,400 seconds = **259,200 rows** |
| Additional anomaly rows | ~600 (spike + card-testing injections) |
| Normal decline rate | ~15% of all transactions |
| Currency | 100% INR |
| Unique card BINs | 9 distinct BINs |

### Normal Decline Codes

| Code | Meaning |
|------|---------|
| `05_Do_Not_Honor` | Generic issuer refusal |
| `51_Insufficient_Funds` | Cardholder account balance too low |
| `14_Invalid_Card_Number` | Card number fails Luhn check |
| `54_Expired_Card` | Card past its expiry date |
| `57_Transaction_Not_Permitted` | Card not allowed for this merchant type |

### Slide Talking Point
> *"At 3 transactions per second sustained over 24 hours, the transactions table contains over a quarter million payment events. This volume is representative of a mid-size Indian payment gateway and ensures the AI agent must distinguish genuine anomalies from background noise — exactly the challenge in production."*

---

## 4. Dataset 3 — `webhook_logs.csv`

### Purpose
Operational log of every webhook delivery attempt made to merchant systems after a payment event. Every row in `transactions.csv` has exactly one corresponding row here, linked by `transaction_id`.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `log_id` | string | Unique log entry ID (`WH-00000000` sequential) |
| `transaction_id` | string | Foreign key → `transactions.csv` |
| `timestamp` | datetime (UTC) | When the webhook delivery was attempted (transaction timestamp + 1–30 s delay) |
| `event_type` | string | `payment.success`, `payment.failed`, or `payment.pending` |
| `http_status` | integer | HTTP response code from the merchant's endpoint |
| `delivery_attempts` | integer | Number of delivery tries (1 for healthy; up to 3 for failures) |
| `latency_ms` | integer | Round-trip delivery latency in milliseconds |

### Key Statistics

| Metric | Value |
|--------|-------|
| Total rows | **1-to-1 with transactions** (~259,800) |
| Healthy deliveries | HTTP 200 (approx. 66% under normal conditions) |
| Baseline latency | 50 – 500 ms |
| Webhook event types | 3 (`payment.success`, `payment.failed`, `payment.pending`) |

### Slide Talking Point
> *"The webhook log captures the delivery reliability of the gateway's notification system. Merchants depend on these events to update their own order management and reconciliation systems. When webhooks fail silently, merchants have no way to know a payment succeeded — a critical operations gap that the AI agent is designed to detect and diagnose."*

---

## 5. Injected Anomalies — AI Agent Test Cases

Six deliberately crafted anomalies are woven into the dataset. They are realistic, time-bounded, and designed to test a different diagnostic capability of the AI agent.

---

### Anomaly 1 — Sudden Decline Spike (Risk Block)

| Attribute | Detail |
|-----------|--------|
| **Affected merchant** | `merchant_id_1` |
| **Symptom** | 50 consecutive `DECLINED` transactions in a **10-minute window** |
| **Decline code** | `93_Risk_Block` |
| **Time anchor** | 6 hours before data generation |
| **What it tests** | Can the agent detect an abnormally high decline rate concentrated in time for a single merchant? |

**Slide Talking Point:**  
> *"This simulates a scenario where the card network's fraud engine has temporarily blocked a card BIN or risk category associated with a merchant. The agent must correlate the timestamp concentration and the specific decline code to conclude this is an external risk block, not a merchant configuration error."*

---

### Anomaly 2 — Unauthorized Webhook Delivery (401 Error)

| Attribute | Detail |
|-----------|--------|
| **Affected merchant** | `merchant_id_2` |
| **Symptom** | All webhook deliveries in the **last 2 hours** return `HTTP 401 Unauthorized` |
| **HTTP status** | `401` |
| **What it tests** | Can the agent identify that a merchant's webhook secret/token has expired or been rotated, causing authentication failures? |

**Slide Talking Point:**  
> *"A 401 on webhook delivery means the gateway's request is being rejected by the merchant's server — typically because the shared secret used to authenticate the payload has expired or been changed. Without diagnosis, the merchant simply stops receiving payment notifications and their reconciliation goes dark."*

---

### Anomaly 3 — Card Testing Attack

| Attribute | Detail |
|-----------|--------|
| **Affected merchant** | `merchant_id_3` |
| **Symptom** | Burst of **100 transactions** in a **5-minute window** with amounts strictly between ₹1 and ₹5 |
| **Decline rate** | ~95% declined |
| **Decline codes** | `14_Invalid_Card_Number`, `54_Expired_Card` |
| **What it tests** | Can the agent recognise the micro-transaction + high-decline-rate + short-burst pattern characteristic of card enumeration attacks? |

**Slide Talking Point:**  
> *"Card testing is a well-known fraud pattern where attackers submit hundreds of small-value transactions to verify stolen card numbers before using them for large purchases. The signature is unmistakable in the data: micro-amounts, a very short time window, and a near-100% decline rate with card-validity error codes."*

---

### Anomaly 4 — Issuer Switch Downtime (BIN-level Outage)

| Attribute | Detail |
|-----------|--------|
| **Affected scope** | **ALL merchants** |
| **Trigger condition** | Card BIN = `411111` within a **2-hour downtime window** |
| **Failure rate** | ≥90% of BIN `411111` transactions declined |
| **Decline code** | `91_Issuer_Switch_Inoperative` |
| **Time anchor** | 8–10 hours before data generation |
| **What it tests** | Can the agent pivot from a merchant-level view to a BIN-level view and identify an issuer infrastructure outage affecting all merchants equally? |

**Slide Talking Point:**  
> *"This is the most complex diagnostic case: the failure is not caused by any single merchant — it affects every merchant on the gateway equally, but only for cards from one specific issuing bank (identified by their BIN prefix). The agent must aggregate across merchants, focus on the BIN dimension, and conclude this is an issuer-side infrastructure problem, not a gateway issue."*

---

### Anomaly 5 — Server Overload (High Volume + 504 Timeouts)

| Attribute | Detail |
|-----------|--------|
| **Affected merchant** | `merchant_id_4` |
| **Symptom** | +500 extra transactions injected in the **20:00–21:00 UTC** window (volume spike) |
| **Webhook impact** | All corresponding webhooks return `HTTP 504 Gateway Timeout` |
| **Latency** | `latency_ms > 5,000 ms` (>5 seconds) for all affected webhooks |
| **What it tests** | Can the agent correlate a transaction volume surge with webhook delivery failures and conclude the merchant's server is being overwhelmed? |

**Slide Talking Point:**  
> *"This simulates a flash-sale or peak-hour event where a merchant's infrastructure cannot keep up with the payment notification volume. The 504 and extreme latency values in the webhook logs — combined with the transaction volume spike — give the agent the evidence it needs to recommend that the merchant scale their webhook processing capacity."*

---

### Anomaly 6 — Silent Webhook Failure (Dropped Notifications)

| Attribute | Detail |
|-----------|--------|
| **Affected merchant** | `merchant_id_5` |
| **Symptom** | Exactly **5 `SUCCESS` transactions** whose corresponding webhooks return `HTTP 500 Internal Server Error` |
| **Delivery attempts** | 3 (exhausted retries) |
| **What it tests** | Can the agent detect a reconciliation gap — where money was collected successfully but the merchant was never notified, so their order management system won't fulfil the order? |

**Slide Talking Point:**  
> *"This is arguably the highest-impact silent failure in payments: the customer is charged, the gateway records a success, but the merchant's system never gets the notification and the order sits unfulfilled. The agent must cross-reference the SUCCESS status in transactions.csv against the 500 status in webhook_logs.csv to surface these dropped notifications."*

---

## 6. Entity-Relationship Summary

```
merchants.csv
└── merchant_id  (PK)
        │
        │ 1 : many
        ▼
transactions.csv
├── transaction_id  (PK)
├── merchant_id     (FK → merchants)
├── card_bin        (links to issuer BIN registry)
└── ...
        │
        │ 1 : 1
        ▼
webhook_logs.csv
├── log_id          (PK)
├── transaction_id  (FK → transactions)
└── ...
```

---

## 7. Why Synthetic Data?

| Concern | How it is addressed |
|---------|---------------------|
| **Privacy** | No real customer names, card numbers, or financial data — all values are Faker-generated |
| **Reproducibility** | Fixed random seed (`42`) produces identical data on every run |
| **Anomaly ground truth** | Anomalies are precisely injected, so evaluation of agent accuracy is exact (no labelling ambiguity) |
| **Scale realism** | 3 TPS × 24 h = 259,200 rows — representative of a mid-size Indian gateway |
| **Locale realism** | Indian business names, INR currency, and Indian MCC distribution |

---

## 8. How to Regenerate the Data

```bash
# From the project root
python data/telemetry_generator.py
```

Output files are written to `data/output/`:

```
data/output/
├── merchants.csv        (25 rows)
├── transactions.csv     (~259,800 rows)
└── webhook_logs.csv     (~259,800 rows)
```

> ⚠️ **Note:** When loading `transactions.csv` in Python, always use the
> provided helper to preserve `card_bin` as a string:
> ```python
> from data.telemetry_generator import read_transactions_csv
> txn = read_transactions_csv("data/output/transactions.csv")
> ```

---

*Generated by `data/telemetry_generator.py` · Intelligent Merchant Support AI Agent · M.Tech Project*
