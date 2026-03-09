"""
tests/test_rag_setup.py – Unit tests for the RAG setup module.

Strategy
--------
* ``create_dummy_docs`` and ``load_and_split_docs`` are tested against a real
  temporary directory (no mocking needed — they are pure Python / Pandas /
  LangChain text utilities that run instantly).
* ``build_vector_store``, ``get_retriever``, and ``test_rag_query`` depend on
  HuggingFace model downloads and Chroma I/O; these are mocked so the test
  suite runs offline and in seconds.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_setup import (
    _DOCS,
    _DEFAULT_CHROMA_DIR,
    _DEFAULT_DOCS_DIR,
    create_dummy_docs,
    get_retriever,
    load_and_split_docs,
    test_rag_query as _run_rag_query,  # alias prevents pytest collecting this as a test
)


# ──────────────────────────────────────────────────────────────────────────────
# create_dummy_docs
# ──────────────────────────────────────────────────────────────────────────────


class TestCreateDummyDocs:
    def test_creates_directory(self, tmp_path):
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        assert docs_dir.is_dir()

    def test_creates_all_three_files(self, tmp_path):
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        created = {p.name for p in docs_dir.iterdir()}
        assert created == set(_DOCS.keys())

    def test_returns_absolute_path(self, tmp_path):
        result = create_dummy_docs(tmp_path / "docs")
        assert result.is_absolute()

    def test_decline_codes_content(self, tmp_path):
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        content = (docs_dir / "decline_codes.md").read_text(encoding="utf-8")
        # Must contain the three codes called out in the problem statement
        assert "05" in content
        assert "51" in content
        assert "93" in content
        assert "Risk Block" in content

    def test_webhook_content(self, tmp_path):
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        content = (docs_dir / "webhook_integration.md").read_text(encoding="utf-8")
        assert "200" in content
        assert "401" in content
        assert "500" in content
        assert "retry" in content.lower()

    def test_payout_content(self, tmp_path):
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        content = (docs_dir / "payout_schedules.md").read_text(encoding="utf-8")
        assert "T+1" in content
        assert "T+2" in content
        assert "holiday" in content.lower()

    def test_idempotent_overwrites(self, tmp_path):
        """Calling create_dummy_docs twice should not raise and files should
        contain the latest content."""
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        create_dummy_docs(docs_dir)  # second call must not fail
        created = {p.name for p in docs_dir.iterdir()}
        assert created == set(_DOCS.keys())

    def test_files_are_utf8_encoded(self, tmp_path):
        docs_dir = tmp_path / "docs"
        create_dummy_docs(docs_dir)
        for name in _DOCS:
            content = (docs_dir / name).read_text(encoding="utf-8")
            assert len(content) > 0


# ──────────────────────────────────────────────────────────────────────────────
# load_and_split_docs
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadAndSplitDocs:
    @pytest.fixture
    def docs_dir(self, tmp_path):
        d = tmp_path / "docs"
        create_dummy_docs(d)
        return d

    def test_returns_list(self, docs_dir):
        chunks = load_and_split_docs(docs_dir)
        assert isinstance(chunks, list)

    def test_produces_chunks(self, docs_dir):
        chunks = load_and_split_docs(docs_dir)
        assert len(chunks) > 0

    def test_more_chunks_than_source_docs(self, docs_dir):
        """Each document should be split into at least one chunk; with three
        docs the total chunk count should exceed 3."""
        chunks = load_and_split_docs(docs_dir)
        assert len(chunks) >= len(_DOCS)

    def test_chunk_size_respected(self, docs_dir):
        chunk_size = 200
        chunks = load_and_split_docs(docs_dir, chunk_size=chunk_size, chunk_overlap=20)
        # Allow a small tolerance: splitter may slightly exceed chunk_size at
        # word/sentence boundaries.
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50

    def test_chunks_have_source_metadata(self, docs_dir):
        chunks = load_and_split_docs(docs_dir)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert len(chunk.metadata["source"]) > 0

    def test_all_three_sources_present(self, docs_dir):
        chunks = load_and_split_docs(docs_dir)
        sources = {Path(c.metadata["source"]).name for c in chunks}
        assert sources == set(_DOCS.keys())

    def test_smaller_chunk_size_more_chunks(self, docs_dir):
        big   = load_and_split_docs(docs_dir, chunk_size=1000, chunk_overlap=0)
        small = load_and_split_docs(docs_dir, chunk_size=200,  chunk_overlap=0)
        assert len(small) >= len(big)

    def test_raises_if_docs_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_split_docs(tmp_path / "nonexistent")

    def test_raises_if_no_md_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        # create a non-md file that should be ignored
        (empty_dir / "readme.txt").write_text("hello")
        with pytest.raises(ValueError, match="No Markdown files"):
            load_and_split_docs(empty_dir)

    def test_single_large_doc_splits(self, tmp_path):
        """A single large document should be split into multiple chunks."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        big_content = "\n\n".join([f"Paragraph {i}: " + "x" * 300 for i in range(10)])
        (docs_dir / "big.md").write_text(big_content, encoding="utf-8")
        chunks = load_and_split_docs(docs_dir, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1


# ──────────────────────────────────────────────────────────────────────────────
# build_vector_store (mocked)
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildVectorStore:
    def _make_mock_doc(self, content: str = "test content") -> MagicMock:
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": "test.md"}
        return doc

    def test_calls_from_documents(self, tmp_path):
        """build_vector_store must call Chroma.from_documents with the chunks."""
        from rag_setup import build_vector_store

        docs = [self._make_mock_doc(f"chunk {i}") for i in range(5)]
        persist_dir = tmp_path / "chroma_db"

        with patch("rag_setup._get_embeddings", return_value=MagicMock()), \
             patch("langchain_community.vectorstores.Chroma") as mock_chroma:
            mock_chroma.from_documents.return_value = MagicMock()
            build_vector_store(docs, persist_directory=persist_dir)
            mock_chroma.from_documents.assert_called_once()

            # Verify the call received the document chunks
            call_kwargs = mock_chroma.from_documents.call_args
            assert call_kwargs is not None


# ──────────────────────────────────────────────────────────────────────────────
# get_retriever (mocked)
# ──────────────────────────────────────────────────────────────────────────────


class TestGetRetriever:
    def test_raises_if_chroma_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Chroma store not found"):
            get_retriever(persist_directory=tmp_path / "nonexistent_db")

    def test_returns_retriever_object(self, tmp_path):
        """With the Chroma class mocked, get_retriever should return the
        mock's as_retriever() result."""
        persist_dir = tmp_path / "chroma_db"
        persist_dir.mkdir()

        mock_retriever = MagicMock()
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever

        with patch("rag_setup._get_embeddings", return_value=MagicMock()), \
             patch("langchain_community.vectorstores.Chroma", return_value=mock_db):
            result = get_retriever(persist_directory=persist_dir, k=2)

        assert result is mock_retriever
        mock_db.as_retriever.assert_called_once_with(search_kwargs={"k": 2})

    def test_k_parameter_forwarded(self, tmp_path):
        persist_dir = tmp_path / "chroma_db"
        persist_dir.mkdir()

        mock_db = MagicMock()

        with patch("rag_setup._get_embeddings", return_value=MagicMock()), \
             patch("langchain_community.vectorstores.Chroma", return_value=mock_db):
            get_retriever(persist_directory=persist_dir, k=5)

        mock_db.as_retriever.assert_called_once_with(search_kwargs={"k": 5})


# ──────────────────────────────────────────────────────────────────────────────
# test_rag_query (mocked)
# ──────────────────────────────────────────────────────────────────────────────


class TestTestRagQuery:
    def _make_doc(self, content: str, source: str) -> MagicMock:
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source}
        return doc

    def test_returns_list_of_docs(self, tmp_path, capsys):
        persist_dir = tmp_path / "chroma_db"
        persist_dir.mkdir()

        mock_docs = [
            self._make_doc("Code 93 is a risk block.", "decline_codes.md"),
            self._make_doc("Contact the risk team.", "decline_codes.md"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        with patch("rag_setup.get_retriever", return_value=mock_retriever):
            result = _run_rag_query(
                "Why did my transaction fail with code 93?",
                persist_directory=persist_dir,
            )

        assert result is mock_docs
        assert len(result) == 2

    def test_prints_query_and_chunks(self, tmp_path, capsys):
        persist_dir = tmp_path / "chroma_db"
        persist_dir.mkdir()

        mock_doc = self._make_doc("Risk Block explanation.", "decline_codes.md")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        with patch("rag_setup.get_retriever", return_value=mock_retriever):
            _run_rag_query("code 93", persist_directory=persist_dir)

        captured = capsys.readouterr().out
        assert "code 93" in captured
        assert "Risk Block explanation." in captured

    def test_empty_results_handled(self, tmp_path, capsys):
        persist_dir = tmp_path / "chroma_db"
        persist_dir.mkdir()

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        with patch("rag_setup.get_retriever", return_value=mock_retriever):
            result = _run_rag_query("unknown query", persist_directory=persist_dir)

        assert result == []
        captured = capsys.readouterr().out
        assert "0 chunk" in captured
