from types import SimpleNamespace

import pytest

from rag_indexer import indexer as module
from rag_indexer.config import Settings
from rag_indexer.indexer import Indexer


def settings(**overrides):
    values = dict(
        postgres_dsn="postgresql://rag:rag@localhost/rag",
        qdrant_url="http://qdrant:6333",
        collection="rag_chunks",
        dense_model="dense-model",
        sparse_model="sparse-model",
        batch_size=2,
    )
    values.update(overrides)
    return Settings(**values)


class FakeEmbedding:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return iter(self.vectors)


class FakeQdrant:
    def __init__(self, exists=False, embedding_size=384):
        self.exists = exists
        self.embedding_size = embedding_size
        self.created = []
        self.payload_indexes = []
        self.upserts = []
        self.deletes = []

    def collection_exists(self, name):
        return self.exists

    def get_embedding_size(self, model):
        return self.embedding_size

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def create_payload_index(self, collection, field, schema):
        self.payload_indexes.append((collection, field, schema))

    def upsert(self, collection, points, wait):
        self.upserts.append((collection, points, wait))

    def delete(self, collection, points_selector, wait):
        self.deletes.append((collection, points_selector, wait))


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, execute_results=None):
        self.execute_results = list(execute_results or [])
        self.calls = []
        self.commits = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self.execute_results:
            return FakeResult(self.execute_results.pop(0))
        return FakeResult([])

    def commit(self):
        self.commits += 1


class FakeConnectionContext:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


def make_indexer():
    obj = Indexer.__new__(Indexer)
    obj.settings = settings()
    obj.qdrant = FakeQdrant()
    return obj


def row(chunk_id=1, tenant="t1", text="hello"):
    return {
        "tenant_id": tenant,
        "id": chunk_id,
        "user_id": "u1",
        "source_name": "mailbox.pst",
        "parent_kind": "message",
        "parent_id": "m1",
        "ordinal": 0,
        "start_char": 0,
        "end_char": len(text),
        "text": text,
    }


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RAG_POSTGRES_DSN", "postgresql://example")
    monkeypatch.setenv("RAG_INDEX_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_OUTBOX_LOCK_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("RAG_OUTBOX_RETRY_BASE_SECONDS", "3")
    monkeypatch.setenv("RAG_OUTBOX_RETRY_MAX_SECONDS", "90")

    s = Settings.from_env()

    assert s.postgres_dsn == "postgresql://example"
    assert s.batch_size == 32
    assert s.outbox_lock_timeout_seconds == 60
    assert s.outbox_retry_base_seconds == 3
    assert s.outbox_retry_max_seconds == 90


def test_ensure_collection_uses_model_embedding_size(monkeypatch):
    fake_qdrant = FakeQdrant(exists=False, embedding_size=768)

    class FakeDense:
        def __init__(self, **kwargs):
            assert kwargs == {"model_name": "dense-model", "lazy_load": True}

    class FakeSparse:
        def __init__(self, **kwargs):
            assert kwargs == {"model_name": "sparse-model", "lazy_load": True}

    monkeypatch.setattr(module, "QdrantClient", lambda **kwargs: fake_qdrant)
    monkeypatch.setattr(module, "TextEmbedding", FakeDense)
    monkeypatch.setattr(module, "SparseTextEmbedding", FakeSparse)

    Indexer(settings())

    assert fake_qdrant.created[0]["vectors_config"]["dense"].size == 768
    assert {x[1] for x in fake_qdrant.payload_indexes} == {
        "tenant_id", "source_name", "user_id"
    }


def test_index_rows_creates_tenant_scoped_point_payload():
    obj = make_indexer()
    obj.dense = FakeEmbedding([SimpleNamespace(tolist=lambda: [0.1, 0.2])])
    obj.sparse = FakeEmbedding([
        SimpleNamespace(
            indices=SimpleNamespace(tolist=lambda: [1, 3]),
            values=SimpleNamespace(tolist=lambda: [0.4, 0.7]),
        )
    ])

    assert obj.index_rows([row()]) == 1

    point = obj.qdrant.upserts[0][1][0]
    assert point.id == "t1:1"
    assert point.payload["tenant_id"] == "t1"
    assert point.payload["chunk_id"] == 1
    assert point.payload["text"] == "hello"


def test_index_rows_is_noop_for_empty_rows():
    obj = make_indexer()
    obj.dense = FakeEmbedding([])
    obj.sparse = FakeEmbedding([])

    assert obj.index_rows([]) == 0
    assert obj.qdrant.upserts == []


def test_index_rows_rejects_embedding_count_mismatch():
    obj = make_indexer()
    obj.dense = FakeEmbedding([])
    obj.sparse = FakeEmbedding([])

    with pytest.raises(RuntimeError, match="embedding count"):
        obj.index_rows([row()])


def test_all_chunks_uses_keyset_pagination_not_offset():
    obj = make_indexer()
    conn = FakeConn([[row(2, "t1"), row(3, "t1")]])

    rows = obj._all_chunks(conn, "t1", 2, "t1", 1)

    assert len(rows) == 2
    sql, params = conn.calls[0]
    assert "OFFSET" not in sql.upper()
    assert "(tenant_id, id) > (%s, %s)" in sql
    assert params == ["t1", "t1", 1, 2]


def test_claim_events_uses_skip_locked_and_increments_attempts():
    obj = make_indexer()
    conn = FakeConn([[{"id": 7, "attempts": 1}]])

    rows = obj._claim_events(conn, 10)

    assert rows[0]["id"] == 7
    sql, params = conn.calls[0]
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "attempts = o.attempts + 1" in sql
    assert params == (300, 10)


def test_run_once_marks_successful_event_processed(monkeypatch):
    event = {
        "id": 7,
        "tenant_id": "t1",
        "event_type": "chunks_replaced",
        "aggregate_type": "message",
        "aggregate_id": "m1",
        "source_name": "mailbox.pst",
        "attempts": 1,
    }
    conn = FakeConn([[event]])

    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *args, **kwargs: FakeConnectionContext(conn),
    )

    obj = make_indexer()
    obj._process_event = lambda conn, event: None

    assert obj.run_once(1) == 1
    assert conn.commits == 2
    assert any("processed_at = now()" in sql for sql, _ in conn.calls)


def test_run_once_retries_failed_event_with_backoff(monkeypatch):
    event = {
        "id": 8,
        "tenant_id": "t1",
        "event_type": "chunks_replaced",
        "aggregate_type": "message",
        "aggregate_id": "m1",
        "source_name": "mailbox.pst",
        "attempts": 2,
    }
    conn = FakeConn([[event]])

    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *args, **kwargs: FakeConnectionContext(conn),
    )

    obj = make_indexer()
    obj._process_event = lambda conn, event: (_ for _ in ()).throw(
        RuntimeError("qdrant unavailable")
    )

    assert obj.run_once(1) == 0
    assert conn.commits == 2

    failure = [
        (sql, params)
        for sql, params in conn.calls
        if "last_error = %s" in sql
    ][0]
    assert failure[1] == (10, "qdrant unavailable", 8)


def test_run_once_does_not_ack_unknown_event(monkeypatch):
    event = {
        "id": 9,
        "tenant_id": "t1",
        "event_type": "unknown",
        "aggregate_type": "message",
        "aggregate_id": "m1",
        "source_name": "mailbox.pst",
        "attempts": 1,
    }
    conn = FakeConn([[event]])

    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *args, **kwargs: FakeConnectionContext(conn),
    )

    obj = make_indexer()
    assert obj.run_once(1) == 0
    assert any("last_error = %s" in sql for sql, _ in conn.calls)
    assert not any("processed_at = now()" in sql for sql, _ in conn.calls)


def test_run_once_source_cleared_calls_delete(monkeypatch):
    event = {
        "id": 10,
        "tenant_id": "t1",
        "event_type": "source_cleared",
        "aggregate_type": None,
        "aggregate_id": None,
        "source_name": "mailbox.pst",
        "attempts": 1,
    }
    conn = FakeConn([[event]])

    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *args, **kwargs: FakeConnectionContext(conn),
    )

    obj = make_indexer()
    deleted = []
    obj._delete_source = lambda tenant, source: deleted.append((tenant, source))

    assert obj.run_once(1) == 1
    assert deleted == [("t1", "mailbox.pst")]


def test_retry_backoff_is_capped():
    obj = make_indexer()
    obj.settings = settings(outbox_retry_base_seconds=5, outbox_retry_max_seconds=30)
    conn = FakeConn()

    obj._mark_failed(conn, 1, RuntimeError("boom"), attempts=10)

    sql, params = conn.calls[0]
    assert "available_at" in sql
    assert params[0] == 30
