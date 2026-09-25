from __future__ import annotations

from dataclasses import dataclass
import time

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from fastembed import SparseTextEmbedding, TextEmbedding

from .config import Settings


@dataclass
class Indexer:
    settings: Settings

    def __post_init__(self):
        self.qdrant = QdrantClient(url=self.settings.qdrant_url)
        self.dense = TextEmbedding(model_name=self.settings.dense_model, lazy_load=True)
        self.sparse = SparseTextEmbedding(model_name=self.settings.sparse_model, lazy_load=True)
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.qdrant.collection_exists(self.settings.collection):
            dense_size = self.qdrant.get_embedding_size(self.settings.dense_model)
            self.qdrant.create_collection(
                collection_name=self.settings.collection,
                vectors_config={
                    "dense": VectorParams(size=dense_size, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                        modifier=Modifier.IDF,
                    )
                },
            )
        for field in ("tenant_id", "source_name", "user_id"):
            self.qdrant.create_payload_index(
                self.settings.collection, field, "keyword"
            )

    def _chunks_for_parent(self, conn, tenant_id: str, parent_kind: str, parent_id: str):
        return conn.execute(
            """
            SELECT tenant_id, id, user_id, source_name, parent_kind, parent_id,
                   ordinal, start_char, end_char, text
            FROM chunks
            WHERE tenant_id = %s AND parent_kind = %s AND parent_id = %s
              AND text <> ''
            ORDER BY ordinal, id
            """,
            (tenant_id, parent_kind, parent_id),
        ).fetchall()

    def _all_chunks(
        self,
        conn,
        tenant_id: str | None,
        limit: int,
        after_tenant: str | None = None,
        after_id: int | None = None,
    ):
        sql = """
            SELECT tenant_id, id, user_id, source_name, parent_kind, parent_id,
                   ordinal, start_char, end_char, text
            FROM chunks
            WHERE text <> ''
        """
        params: list[object] = []
        if tenant_id:
            sql += " AND tenant_id = %s"
            params.append(tenant_id)
        if after_tenant is not None and after_id is not None:
            sql += " AND (tenant_id, id) > (%s, %s)"
            params.extend([after_tenant, after_id])
        sql += " ORDER BY tenant_id, id LIMIT %s"
        params.append(limit)
        return conn.execute(sql, params).fetchall()

    def index_rows(self, rows):
        if not rows:
            return 0

        texts = [r["text"] for r in rows]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        if len(dense) != len(rows) or len(sparse) != len(rows):
            raise RuntimeError("embedding count does not match row count")

        points = []
        for row, d, s in zip(rows, dense, sparse):
            points.append(
                PointStruct(
                    id=f'{row["tenant_id"]}:{row["id"]}',
                    vector={
                        "dense": d.tolist(),
                        "sparse": SparseVector(
                            indices=s.indices.tolist(), values=s.values.tolist()
                        ),
                    },
                    payload={
                        "tenant_id": row["tenant_id"],
                        "user_id": row["user_id"],
                        "chunk_id": row["id"],
                        "source_name": row["source_name"],
                        "parent_kind": row["parent_kind"],
                        "parent_id": row["parent_id"],
                        "ordinal": row["ordinal"],
                        "text": row["text"],
                    },
                )
            )

        self.qdrant.upsert(self.settings.collection, points=points, wait=True)
        return len(points)

    def _delete_source(self, tenant_id: str, source_name: str):
        self.qdrant.delete(
            self.settings.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id", match=MatchValue(value=tenant_id)
                    ),
                    FieldCondition(
                        key="source_name", match=MatchValue(value=source_name)
                    ),
                ]
            ),
            wait=True,
        )

    def _claim_events(self, conn, limit: int):
        return conn.execute(
            """
            WITH candidates AS (
                SELECT id
                FROM rag_outbox
                WHERE processed_at IS NULL
                  AND available_at <= now()
                  AND (
                    locked_at IS NULL
                    OR locked_at < now() - (%s * interval '1 second')
                  )
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE rag_outbox AS o
            SET locked_at = now(),
                attempts = o.attempts + 1
            FROM candidates
            WHERE o.id = candidates.id
            RETURNING o.id, o.tenant_id, o.event_type, o.aggregate_type,
                      o.aggregate_id, o.source_name, o.attempts
            """,
            (self.settings.outbox_lock_timeout_seconds, limit),
        ).fetchall()

    def _mark_processed(self, conn, event_id: int):
        conn.execute(
            """
            UPDATE rag_outbox
            SET processed_at = now(), locked_at = NULL, last_error = NULL
            WHERE id = %s AND processed_at IS NULL
            """,
            (event_id,),
        )

    def _mark_failed(self, conn, event_id: int, error: Exception, attempts: int):
        delay = min(
            self.settings.outbox_retry_max_seconds,
            self.settings.outbox_retry_base_seconds * (2 ** max(attempts, 0)),
        )
        conn.execute(
            """
            UPDATE rag_outbox
            SET locked_at = NULL,
                available_at = now() + (%s * interval '1 second'),
                last_error = %s
            WHERE id = %s AND processed_at IS NULL
            """,
            (delay, str(error)[:4000], event_id),
        )

    def _process_event(self, conn, event):
        if event["event_type"] == "chunks_replaced":
            rows = self._chunks_for_parent(
                conn,
                event["tenant_id"],
                event["aggregate_type"],
                event["aggregate_id"],
            )
            self.index_rows(rows)
            return
        if event["event_type"] == "source_cleared":
            self._delete_source(event["tenant_id"], event["source_name"])
            return
        raise ValueError(f"unsupported outbox event type: {event["event_type"]}")

    def run_once(self, limit: int | None = None):
        limit = limit or self.settings.batch_size
        processed = 0

        with psycopg.connect(self.settings.postgres_dsn, row_factory=dict_row) as conn:
            events = self._claim_events(conn, limit)
            conn.commit()

            for event in events:
                try:
                    self._process_event(conn, event)
                except Exception as exc:
                    self._mark_failed(conn, event["id"], exc, event["attempts"])
                    conn.commit()
                    continue

                self._mark_processed(conn, event["id"])
                conn.commit()
                processed += 1

        return processed

    def rebuild(self, tenant_id: str | None = None):
        total = 0
        after_tenant = None
        after_id = None

        with psycopg.connect(self.settings.postgres_dsn, row_factory=dict_row) as conn:
            while True:
                rows = self._all_chunks(
                    conn,
                    tenant_id,
                    self.settings.batch_size,
                    after_tenant,
                    after_id,
                )
                if not rows:
                    break

                total += self.index_rows(rows)
                last = rows[-1]
                after_tenant = last["tenant_id"]
                after_id = last["id"]

                if len(rows) < self.settings.batch_size:
                    break

        return total


def run_forever(settings: Settings):
    indexer = Indexer(settings)
    while True:
        indexed = indexer.run_once(settings.batch_size)
        if indexed == 0:
            time.sleep(1)
