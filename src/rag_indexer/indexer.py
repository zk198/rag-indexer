from __future__ import annotations

from dataclasses import dataclass
import time

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, Filter, FieldCondition, MatchValue, Modifier,
)
from fastembed import TextEmbedding, SparseTextEmbedding

from .config import Settings


@dataclass
class Indexer:
    settings: Settings

    def __post_init__(self):
        self.qdrant = QdrantClient(url=self.settings.qdrant_url)
        self.dense = TextEmbedding(self.settings.dense_model)
        self.sparse = SparseTextEmbedding(self.settings.sparse_model)
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.qdrant.collection_exists(self.settings.collection):
            self.qdrant.create_collection(
                collection_name=self.settings.collection,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False), modifier=Modifier.IDF)},
            )
        self.qdrant.create_payload_index(self.settings.collection, "tenant_id", "keyword")
        self.qdrant.create_payload_index(self.settings.collection, "source_name", "keyword")
        self.qdrant.create_payload_index(self.settings.collection, "user_id", "keyword")

    def _chunks_for_parent(self, conn, tenant_id: str, parent_kind: str, parent_id: str):
        return conn.execute(
            """
            SELECT tenant_id, id, user_id, source_name, parent_kind, parent_id,
                   ordinal, start_char, end_char, text
            FROM chunks
            WHERE tenant_id = %s AND parent_kind = %s AND parent_id = %s
            ORDER BY ordinal, id
            """,
            (tenant_id, parent_kind, parent_id),
        ).fetchall()

    def _all_chunks(self, conn, tenant_id: str | None, limit: int, offset: int):
        sql = """
            SELECT tenant_id, id, user_id, source_name, parent_kind, parent_id,
                   ordinal, start_char, end_char, text
            FROM chunks
            WHERE text <> ''
        """
        params = []
        if tenant_id:
            sql += " AND tenant_id = %s"
            params.append(tenant_id)
        sql += " ORDER BY tenant_id, id OFFSET %s LIMIT %s"
        params.extend([offset, limit])
        return conn.execute(sql, params).fetchall()

    def index_rows(self, rows):
        if not rows:
            return 0
        texts = [r["text"] for r in rows]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        points = []
        for row, d, s in zip(rows, dense, sparse):
            points.append(PointStruct(
                id=f'{row["tenant_id"]}:{row["id"]}',
                vector={
                    "dense": d.tolist(),
                    "sparse": SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload={
                    "tenant_id": row["tenant_id"], "user_id": row["user_id"],
                    "chunk_id": row["id"], "source_name": row["source_name"],
                    "parent_kind": row["parent_kind"], "parent_id": row["parent_id"],
                    "ordinal": row["ordinal"], "text": row["text"],
                },
            ))
        self.qdrant.upsert(self.settings.collection, points=points, wait=True)
        return len(points)

    def _delete_source(self, tenant_id: str, source_name: str):
        self.qdrant.delete(
            self.settings.collection,
            points_selector=Filter(must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="source_name", match=MatchValue(value=source_name)),
            ]),
            wait=True,
        )

    def run_once(self, limit=64):
        processed = 0
        with psycopg.connect(self.settings.postgres_dsn, row_factory=dict_row) as conn:
            events = conn.execute(
                """
                SELECT id, tenant_id, event_type, aggregate_type, aggregate_id, source_name
                FROM rag_outbox
                WHERE processed_at IS NULL AND available_at <= now()
                ORDER BY id
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            for event in events:
                if event["event_type"] == "chunks_replaced":
                    rows = self._chunks_for_parent(
                        conn, event["tenant_id"], event["aggregate_type"], event["aggregate_id"]
                    )
                    self.index_rows(rows)
                elif event["event_type"] == "source_cleared":
                    self._delete_source(event["tenant_id"], event["source_name"])
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE rag_outbox SET processed_at = now(), locked_at = NULL WHERE id = %s",
                        (event["id"],),
                    )
                processed += 1
            return processed

    def rebuild(self, tenant_id: str | None = None):
        offset = 0
        total = 0
        with psycopg.connect(self.settings.postgres_dsn, row_factory=dict_row) as conn:
            while True:
                rows = self._all_chunks(conn, tenant_id, self.settings.batch_size, offset)
                if not rows:
                    break
                total += self.index_rows(rows)
                offset += len(rows)
                if len(rows) < self.settings.batch_size:
                    break
        return total


def run_forever(settings: Settings):
    indexer = Indexer(settings)
    while True:
        indexed = indexer.run_once(settings.batch_size)
        if indexed == 0:
            time.sleep(1)
