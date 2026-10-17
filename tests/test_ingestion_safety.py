"""T19: ingestion never drops or rebuilds the existing collection and enforces its budget."""

from pathlib import Path

import fitz
import pytest

import main_pipeline
from main_pipeline import IngestionCompatibilityError, IngestionLimitError, MultiPDFPipeline


class FakeField:
    def __init__(self, name, dim=None):
        self.name = name
        self.params = {"dim": dim} if dim else {}


class FakeCollection:
    def __init__(self, dim=1536, has_index=True):
        self.schema = type("S", (), {"fields": [FakeField("id"), FakeField("embedding", dim)]})()
        self._has_index = has_index
        self.calls = []

    def has_index(self):
        return self._has_index

    def create_index(self, **kw):
        self.calls.append("create_index")

    def insert(self, data):
        self.calls.append(("insert", len(data[0])))

    def load(self):
        self.calls.append("load")

    def drop(self):  # must never be called
        self.calls.append("drop")


class FakeMilvusDB:
    collection = None

    def __init__(self):
        pass

    def create_collection(self, collection_name="ai_research_assistant"):
        return FakeMilvusDB.collection


def make_pdf(path: Path, pages=2, words=300):
    doc = fitz.open()
    for p in range(pages):
        page = doc.new_page()
        text = " ".join(f"word{p}_{i}" for i in range(words))
        page.insert_textbox(fitz.Rect(36, 36, 560, 800), text, fontsize=6)
    doc.save(path)
    return path


@pytest.fixture
def pdf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # extracted tables/images land in tmp, not the repo
    return make_pdf(tmp_path / "sample.pdf")


@pytest.fixture
def fake_db(monkeypatch):
    FakeMilvusDB.collection = FakeCollection()
    monkeypatch.setattr(main_pipeline, "MilvusDB", FakeMilvusDB)
    return FakeMilvusDB


def test_existing_collection_and_index_are_preserved(pdf, fake_db, fake_mesh, settings):
    assert MultiPDFPipeline(settings=settings).run(single_pdf_path=pdf) is True
    calls = fake_db.collection.calls
    assert "drop" not in calls and "create_index" not in calls
    assert any(c[0] == "insert" for c in calls if isinstance(c, tuple))


def test_embedding_requests_are_batched_within_ingestion_budget(pdf, fake_db, fake_mesh, settings):
    p = MultiPDFPipeline(settings=settings)
    p.run(single_pdf_path=pdf)
    snap = p.budget.snapshot()
    assert 1 <= snap.mesh_embedding_calls <= 50
    assert max(fake_mesh.embeddings.batches) <= 64
    assert snap.mesh_chat_calls == 0  # ingestion budget is embedding-only


def test_chunk_limit_rejects_upload_before_any_embedding(pdf, fake_db, fake_mesh, use_settings):
    s = use_settings(MAX_INGEST_EMBEDDING_CALLS=1, EMBEDDING_BATCH_SIZE=2)
    with pytest.raises(IngestionLimitError, match="Nothing was indexed"):
        MultiPDFPipeline(settings=s).run(single_pdf_path=pdf)
    assert fake_mesh.embeddings.calls == 0
    assert fake_db.collection.calls == []


def test_file_size_limit(pdf, fake_db, fake_mesh, use_settings):
    s = use_settings(MAX_UPLOAD_MB=0.000001)
    with pytest.raises(IngestionLimitError, match="upload limit"):
        MultiPDFPipeline(settings=s).run(single_pdf_path=pdf)
    assert fake_mesh.embeddings.calls == 0


def test_dimension_mismatch_refuses_to_insert(pdf, fake_db, fake_mesh, settings):
    fake_db.collection = FakeCollection(dim=768)
    with pytest.raises(IngestionCompatibilityError):
        MultiPDFPipeline(settings=settings).run(single_pdf_path=pdf)
    assert not any(isinstance(c, tuple) for c in fake_db.collection.calls)


def test_incompatible_flag_refuses_to_mix_vectors(pdf, fake_db, fake_mesh, use_settings):
    s = use_settings(MILVUS_EMBEDDING_COMPAT="incompatible")
    with pytest.raises(IngestionCompatibilityError):
        MultiPDFPipeline(settings=s).run(single_pdf_path=pdf)
    assert fake_mesh.embeddings.calls == 0


def test_ensure_index_creates_only_when_missing():
    from vector_database.indexing import IndexManager

    existing, missing = FakeCollection(has_index=True), FakeCollection(has_index=False)
    assert IndexManager.ensure_hnsw_index(existing) is False and existing.calls == []
    assert IndexManager.ensure_hnsw_index(missing) is True and missing.calls == ["create_index"]
