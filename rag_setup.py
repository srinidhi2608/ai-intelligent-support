"""
rag_setup.py – Knowledge Base module: builds a local RAG (Retrieval-Augmented
               Generation) pipeline for the Intelligent Merchant Support Agent.

This script:
  1. Generates three FinTech Markdown knowledge-base documents under ``docs/``.
  2. Loads and chunks those documents with RecursiveCharacterTextSplitter.
  3. Embeds the chunks with HuggingFace's ``all-MiniLM-L6-v2`` (fully local,
     no API key required).
  4. Persists a Chroma vector store to ``chroma_db/``.
  5. Exposes ``get_retriever()`` and ``test_rag_query()`` helpers so LangChain
     agents can query the knowledge base at runtime.

Run directly to build the KB end-to-end and verify retrieval::

    python rag_setup.py

Typical agent usage::

    from rag_setup import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("What does decline code 93 mean?")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default directory paths ────────────────────────────────────────────────────
_DEFAULT_DOCS_DIR   = Path("docs")
_DEFAULT_CHROMA_DIR = Path("chroma_db")

# ── Dummy FinTech knowledge-base documents ────────────────────────────────────
# Each key is the filename; each value is the full Markdown content.
_DOCS: dict[str, str] = {
    # ── 1. Decline code reference ────────────────────────────────────────────
    "decline_codes.md": """\
# Payment Decline Code Reference

## Overview

Decline codes are ISO 8583 response codes returned by the issuing bank when a
transaction cannot be authorised.  Understanding them helps merchants diagnose
failures and take the appropriate corrective action.

## Common Decline Codes

### Code 05 – Do Not Honor

- **Meaning:** The issuing bank has blocked this transaction without providing a
  specific reason.  Often a catch-all for suspected fraud or account restrictions.
- **Merchant Action:** Ask the cardholder to contact their bank or try an
  alternate payment method.

### Code 14 – Invalid Card Number

- **Meaning:** The card number provided does not pass the Luhn check or is
  unknown to the issuing network.
- **Merchant Action:** Verify the card number with the cardholder and re-enter.

### Code 51 – Insufficient Funds

- **Meaning:** The cardholder's account does not have enough balance to cover
  the transaction amount.
- **Merchant Action:** Request a different card or a reduced amount.

### Code 54 – Expired Card

- **Meaning:** The card's expiry date has passed.
- **Merchant Action:** Request the cardholder's updated card details.

### Code 57 – Transaction Not Permitted

- **Meaning:** The card is not allowed for this merchant category (MCC).
  For example, a corporate card restricted to business travel used at a grocery
  store.
- **Merchant Action:** No action possible; the cardholder must use a different card.

### Code 91 – Issuer Switch Inoperative

- **Meaning:** The issuing bank's network is temporarily unavailable.
- **Merchant Action:** Retry the transaction after a short delay (usually
  resolves within 15–60 minutes).

### Code 93 – Risk Block

- **Meaning:** The transaction has been flagged and blocked by internal fraud
  velocity rules.  This occurs when a card or merchant has exceeded a threshold
  of suspicious activity (e.g., too many declined attempts in a short window).
- **When to Expect It:** A sudden spike of code 93 declines usually indicates
  that the card network's fraud engine has temporarily blocked a specific card
  BIN or risk category associated with the merchant.
- **Merchant Action:** Contact the payment gateway's risk team immediately.
  The block is usually temporary but may require manual review.  Provide the
  affected transaction IDs and the time window for faster resolution.
""",

    # ── 2. Webhook integration guide ─────────────────────────────────────────
    "webhook_integration.md": """\
# Webhook Integration Guide

## What Are Webhooks?

Webhooks are real-time HTTP POST notifications that the payment gateway sends to
your server whenever a payment event occurs.  Unlike polling, webhooks push data
to you instantly, enabling live order fulfilment and reconciliation.

## Supported Events

| Event | Description |
|-------|-------------|
| `payment.success`  | Transaction authorised and captured successfully |
| `payment.failed`   | Transaction declined by the issuer or gateway |
| `payment.pending`  | Transaction is awaiting asynchronous confirmation |
| `payment.refunded` | Refund processed for a previous transaction |

## Expected HTTP Status Codes

Your webhook endpoint must respond with an HTTP status code within **5 seconds**.

| HTTP Code | Meaning | Gateway Action |
|-----------|---------|----------------|
| `200 OK` | Delivery successful | No retry |
| `201 Created` | Delivery successful | No retry |
| `400 Bad Request` | Payload rejected by your server | No retry (client error) |
| `401 Unauthorized` | Authentication failure – webhook secret is wrong or expired | No retry; alert raised |
| `404 Not Found` | Endpoint URL is incorrect | No retry; alert raised |
| `500 Internal Server Error` | Your server crashed | Retried up to 3 times |
| `502 Bad Gateway` | Upstream proxy failure | Retried up to 3 times |
| `503 Service Unavailable` | Server overloaded | Retried up to 3 times |
| `504 Gateway Timeout` | Server took too long | Retried up to 3 times |

## Retry Policy

Webhooks are retried **3 times** if a 5xx error is received, using an
exponential back-off schedule:

1. First retry: **30 seconds** after the initial failure
2. Second retry: **5 minutes** after the first retry
3. Third retry: **30 minutes** after the second retry

If all three retries fail, the event is marked as `DROPPED` and an alert is
generated in the merchant dashboard.

## Securing Webhooks

Every webhook payload is signed with an HMAC-SHA256 signature using your
webhook secret.  The signature is passed in the `X-Gateway-Signature` header.
Always validate this header before processing the payload to prevent spoofing.

## Common Failure Diagnoses

- **401 errors** – Your webhook secret has been rotated.  Update the secret in
  your integration settings and redeploy.
- **504 errors with high latency** – Your server is struggling under load.
  Consider offloading webhook processing to an async queue (e.g., Celery or
  RabbitMQ) so the endpoint returns 200 immediately.
- **500 errors** – Check your application server logs for exceptions.  Verify
  that your database and downstream services are healthy.
""",

    # ── 3. Payout schedules ──────────────────────────────────────────────────
    "payout_schedules.md": """\
# Payout & Settlement Schedules

## Overview

Funds collected from successful transactions are settled to your registered bank
account according to a predefined schedule.  The settlement cycle depends on
your merchant tier and the acquiring bank's clearing timelines.

## Settlement Cycles

### T+1 Settlement

- **Definition:** Funds are transferred to your bank account **one business day**
  after the transaction date (T).
- **Availability:** Available to merchants on the **Premium** and **Enterprise**
  tiers.
- **Cutoff Time:** Transactions captured before **18:00 IST** on business day T
  are included in the T+1 batch.  Transactions captured after 18:00 roll into
  the next day's batch.

### T+2 Settlement

- **Definition:** Funds are transferred **two business days** after the
  transaction date (T).
- **Availability:** Default for **Starter** tier merchants.
- **Cutoff Time:** Transactions captured before **23:59 IST** on day T are
  included in the T+2 batch.

## How Bank Holidays Affect Payouts

Settlement batches are only processed on **bank working days**.  If the
scheduled settlement date falls on a public holiday or bank holiday, the payout
is deferred to the **next available working day**.

### Examples

| Scenario | Scheduled Date | Actual Payout |
|----------|----------------|---------------|
| T+1, holiday on T+1 | Monday → Tuesday | Wednesday |
| T+2, holiday on T+2 | Friday → Monday | Tuesday |
| T+1, transaction on Saturday | Saturday → T+1 = Monday | Monday |

## Holiday Calendar

The platform follows the **RBI (Reserve Bank of India)** holiday schedule for
bank working days.  A full calendar is available in your merchant dashboard
under **Settings → Payout Schedule**.

## Delayed Payouts

If your payout has not arrived by the expected date:

1. Check the **Payouts** section of your dashboard for the batch status.
2. Verify your bank account details are correct and active.
3. Raise a support ticket if the delay exceeds **3 working days** beyond the
   scheduled date.

## Holds and Reserves

- **Rolling Reserve:** A percentage of each transaction may be held for 90 days
  as a risk reserve, particularly for new or high-risk merchants.
- **Payout Hold:** Payouts may be paused during a fraud investigation or when
  your account is under review.
""",
}


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Generate dummy documentation
# ──────────────────────────────────────────────────────────────────────────────


def create_dummy_docs(docs_dir: str | Path = _DEFAULT_DOCS_DIR) -> Path:
    """
    Create the ``docs/`` directory and write three FinTech knowledge-base
    Markdown files into it.

    Files created:

    * ``decline_codes.md``      – ISO 8583 decline code explanations
    * ``webhook_integration.md``– Webhook HTTP statuses and retry policy
    * ``payout_schedules.md``   – T+1 / T+2 settlement and holiday rules

    If the files already exist they are **overwritten** so the content is
    always in sync with the constants defined in this module.

    Parameters
    ----------
    docs_dir:
        Path to the directory where the Markdown files will be written.
        Defaults to ``docs/`` relative to the current working directory.

    Returns
    -------
    Path
        The absolute path to the docs directory.
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    for filename, content in _DOCS.items():
        file_path = docs_path / filename
        file_path.write_text(content, encoding="utf-8")
        logger.info("Written: %s (%d chars)", file_path, len(content))

    logger.info("Dummy docs created in '%s' (%d files)", docs_path, len(_DOCS))
    return docs_path.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Document loading & splitting
# ──────────────────────────────────────────────────────────────────────────────


def load_and_split_docs(
    docs_dir: str | Path = _DEFAULT_DOCS_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Any]:
    """
    Load all Markdown files from *docs_dir* and split them into chunks.

    Uses LangChain's :class:`~langchain_community.document_loaders.DirectoryLoader`
    with :class:`~langchain_community.document_loaders.TextLoader` to read
    every ``*.md`` file, then applies
    :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter` to
    divide the text into overlapping chunks.

    **Why chunking?**  LLM context windows are limited.  Splitting large
    documents into small, overlapping chunks ensures that:

    * Each chunk fits comfortably in the prompt.
    * Sentences split at a boundary are still retrievable via the overlapping
      region of the adjacent chunk.

    Parameters
    ----------
    docs_dir:
        Directory containing ``*.md`` files to load.
    chunk_size:
        Maximum number of characters per chunk (default: 500).
    chunk_overlap:
        Number of characters shared between adjacent chunks (default: 50).
        This prevents losing context at chunk boundaries.

    Returns
    -------
    list[Document]
        A flat list of LangChain :class:`~langchain_core.documents.Document`
        objects, each carrying the chunk text and source-file metadata.

    Raises
    ------
    FileNotFoundError:
        If *docs_dir* does not exist.
    ValueError:
        If no ``*.md`` files are found in *docs_dir*.
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(
            f"docs directory not found: '{docs_path}'. "
            "Call create_dummy_docs() first."
        )

    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(
            f"No Markdown files found in '{docs_path}'. "
            "Ensure create_dummy_docs() has been called."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    logger.info(
        "Loaded %d raw docs → split into %d chunks "
        "(chunk_size=%d, chunk_overlap=%d)",
        len(raw_docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Embedding & vector store
# ──────────────────────────────────────────────────────────────────────────────


def _get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Instantiate a local HuggingFace embedding model.

    Uses ``all-MiniLM-L6-v2`` by default — a lightweight 22 M-parameter
    sentence-transformer that runs on CPU without any API key.

    Parameters
    ----------
    model_name:
        Sentence-Transformers model identifier.

    Returns
    -------
    HuggingFaceEmbeddings
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: list[Any],
    persist_directory: str | Path = _DEFAULT_CHROMA_DIR,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Any:
    """
    Embed *chunks* and persist a Chroma vector store to disk.

    The store is written to *persist_directory* so it survives process
    restarts.  Subsequent calls to :func:`get_retriever` load this directory
    without re-embedding.

    Parameters
    ----------
    chunks:
        List of :class:`~langchain_core.documents.Document` objects produced by
        :func:`load_and_split_docs`.
    persist_directory:
        Directory where Chroma will write its SQLite database and embedding
        index.  Defaults to ``chroma_db/``.
    embedding_model:
        Sentence-Transformers model used for embedding.  Must match the model
        used when :func:`get_retriever` loads the store later.

    Returns
    -------
    Chroma
        A LangChain :class:`~langchain_community.vectorstores.Chroma` instance
        that can be used to create a retriever.
    """
    from langchain_community.vectorstores import Chroma

    embeddings = _get_embeddings(embedding_model)
    persist_dir = str(Path(persist_directory))

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    logger.info(
        "Chroma vector store built with %d chunks, persisted to '%s'",
        len(chunks),
        persist_dir,
    )
    return db


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Retriever
# ──────────────────────────────────────────────────────────────────────────────


def get_retriever(
    persist_directory: str | Path = _DEFAULT_CHROMA_DIR,
    embedding_model: str = "all-MiniLM-L6-v2",
    k: int = 2,
) -> Any:
    """
    Load the persisted Chroma vector store from disk and return a retriever.

    Parameters
    ----------
    persist_directory:
        Path to the directory where the Chroma store was saved by
        :func:`build_vector_store`.
    embedding_model:
        Sentence-Transformers model name — **must** match the model used when
        the store was built.
    k:
        Number of chunks to return per query (default: 2).

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever that can be plugged into a
        ``RetrievalQA`` chain or used standalone with ``retriever.invoke()``.

    Raises
    ------
    FileNotFoundError:
        If *persist_directory* does not exist (i.e. the store has not been
        built yet).
    """
    from langchain_community.vectorstores import Chroma

    persist_dir = Path(persist_directory)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Chroma store not found at '{persist_dir}'. "
            "Run build_vector_store() (or `python rag_setup.py`) first."
        )

    embeddings = _get_embeddings(embedding_model)
    db = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": k})
    logger.info(
        "Retriever loaded from '%s' (k=%d)", persist_dir, k
    )
    return retriever


def test_rag_query(
    query: str,
    persist_directory: str | Path = _DEFAULT_CHROMA_DIR,
    k: int = 2,
) -> list[Any]:
    """
    Run a single retrieval query against the Chroma vector store and print the
    matching document chunks.

    Parameters
    ----------
    query:
        A natural-language question to retrieve context for.
    persist_directory:
        Path to the Chroma vector store directory.
    k:
        Number of chunks to retrieve (default: 2).

    Returns
    -------
    list[Document]
        The retrieved :class:`~langchain_core.documents.Document` objects.
    """
    retriever = get_retriever(persist_directory=persist_directory, k=k)
    results = retriever.invoke(query)

    print(f"\nQuery: {query!r}")
    print(f"Retrieved {len(results)} chunk(s):")
    print("-" * 60)
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n[Chunk {i}] Source: {source}")
        print(doc.page_content)
    print("-" * 60)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stderr,
    )

    print("=" * 60)
    print("  RAG Setup – Intelligent Merchant Support Knowledge Base")
    print("=" * 60)

    # Step 1 – Create the dummy FinTech documentation
    print("\n[1/4] Generating knowledge-base documents...")
    create_dummy_docs()
    print("      ✓  docs/ created with 3 Markdown files")

    # Step 2 – Load and split documents into chunks
    print("\n[2/4] Loading and splitting documents...")
    chunks = load_and_split_docs()
    print(f"      ✓  {len(chunks)} chunks produced")

    # Step 3 – Build the Chroma vector store
    print("\n[3/4] Building Chroma vector store (this may take ~30 s on first run)...")
    build_vector_store(chunks)
    print("      ✓  Vector store persisted to chroma_db/")

    # Step 4 – Test retrieval
    print("\n[4/4] Running test retrieval query...")
    test_rag_query("Why did my transaction fail with code 93?")
    print("\n      ✓  Retrieval test complete")
    print("\nSetup finished. Use get_retriever() in your agents to query the KB.")
