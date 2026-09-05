"""Ingestion pipeline with a fake extractor, a fake database pool and no OpenAI.

Needs the `ingest` extra (LangChain text splitters); skipped otherwise.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

pytest.importorskip("langchain_text_splitters")
pytest.importorskip("langchain_openai")

from agent.models import IngestionConfig  # noqa: E402
from ingestion import ingest as ingest_module  # noqa: E402
from ingestion.chunker import ChunkingConfig, PDFSemanticChunker  # noqa: E402
from ingestion.extract_files import PDFExtractor  # noqa: E402


class FakeExtractor:
    def extract_pdf_content(self, pdf_path):
        return "Alpha paragraph. " * 40 + "\n\n" + "Beta paragraph. " * 40, {
            "source": pdf_path,
            "title": "fake",
            "pages": 1,
            "pictures": 0,
            "tables": 0,
        }


class FakeConn:
    def __init__(self):
        self.rows = []

    @asynccontextmanager
    async def transaction(self):
        yield

    async def fetchrow(self, query, *args):
        self.rows.append(("fetchrow", query.strip().split()[0], args))
        return {"id": "doc-1", "exists": True}

    async def execute(self, query, *args):
        self.rows.append(("execute", query.strip().split()[0], args))


class FakePool:
    def __init__(self):
        self.conn = FakeConn()

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


def test_recursive_chunking_needs_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Building the chunker used to construct OpenAIEmbeddings unconditionally.
    chunker = PDFSemanticChunker(
        ChunkingConfig(chunk_size=200, chunk_overlap=20, use_semantic_splitting=False)
    )
    chunks = chunker.chunk_content("word " * 300, title="t", source="s")
    assert len(chunks) > 1
    assert all(c.metadata["chunk_method"] == "recursive" for c in chunks)
    assert all(c.token_count for c in chunks)


def test_extractor_without_docling_fails_at_construction_not_later(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_docling(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError("no docling")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_docling)
    with pytest.raises(RuntimeError, match=r"\[ingest\] extra"):
        PDFExtractor()


async def test_pipeline_saves_document_and_chunks(monkeypatch, tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    pool = FakePool()
    monkeypatch.setattr(ingest_module, "db_pool", pool)

    async def fake_init():
        return None

    monkeypatch.setattr(ingest_module, "initialize_database", fake_init)
    monkeypatch.setattr(ingest_module, "close_database", fake_init)

    async def fake_schema(path):
        return None

    monkeypatch.setattr(ingest_module, "execute_init_sql", fake_schema)

    pipeline = ingest_module.DocumentIngestionPipeline(
        IngestionConfig(chunk_size=200, chunk_overlap=20, use_semantic_chunking=False),
        documents_folder=str(tmp_path),
        extractor=FakeExtractor(),
    )

    async def fake_embed(chunks, model="m"):
        for c in chunks:
            c.embedding = [0.1, 0.2, 0.3]
        return chunks

    pipeline.aembed_chunks = fake_embed  # type: ignore[method-assign]

    results = await pipeline.ingest_documents()
    assert len(results) == 1 and results[0].chunks_created > 1
    inserts = [r for r in pool.conn.rows if r[1] == "INSERT"]
    assert inserts[0][2][0] == "fake"  # document title
    assert all(r[2][2] == "[0.1,0.2,0.3]" for r in inserts[1:])  # pgvector literal per chunk
