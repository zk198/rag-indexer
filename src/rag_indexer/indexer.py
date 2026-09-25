from __future__ import annotations

from dataclasses import dataclass
import time

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, Filter, FieldCondition, MatchValue,
)
from fastembed import TextEmbedding, SparseTextEmbedding

from .config import Settings

@dataclass
class Indexer:
    settings: Settings

    def __post_init__(self):
        self.db = self.settings.postgres_dsn
        self.qdrant = QdrantClient(url=self.settings.qdrant_url)
        self.dense = TextEmbedding(self.settings.dense_model)
        self.sparse = SparseTextEmbedding(self.settings.sparse_model)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.qdrant.collection_exists(self.settings.collection):
            return
        self.qdrant.create_collection(
            collection_name=self.settings.collection,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
        )
        self.qdrant.create_payload_index(self.settings.collection, "tenant_id", "keyword")

    def _fetch(self, conn, tenant_id: str | None = None, limit: int = 64):
        sql = """
            SELECT c.tenant_id, c.id, c.user_id, c.source_name, c.parent_kind,
                   c.parent_id, c.ordinal, c.start_char, c.end_char, c.text
            FROM chunks c
            LEFT JOIN rag_outbox o
              ON o.tenant_id = c.tenant_id
             AND o.aggregate_id = c.parent_id
             AND o.aggregate_type = c.parent_kind
             AND o.processed_at IS NULL
            WHERE c.text <> ''
        """
        params=[]
        if tenant_id:
            sql += " AND c.tenant_id = %s"
            params.append(tenant_id)
        sql += " ORDER BY c.created_at, c.id LIMIT %s"
        params.append(limit)
        return conn.execute(sql, params).fetchall()

    def index_rows(self, rows):
        if not rows:
            return 0
        texts=[r["text"] for r in rows]
        dense=list(self.dense.embed(texts))
        sparse=list(self.sparse.embed(texts))
        points=[]
        for row,d,s in zip(rows,dense,sparse):
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

    def run_once(self, limit=64):
        with psycopg.connect(self.db, row_factory=dict_row) as conn:
            rows=self._fetch(limit=limit)
            return self.index_rows(rows)

    def rebuild(self, tenant_id: str | None = None):
        with psycopg.connect(self.db, row_factory=dict_row) as conn:
            offset=0
            total=0
            while True:
                rows=self._fetch(conn, tenant_id, self.settings.batch_size)
                if not rows: break
                total += self.index_rows(rows)
                offset += len(rows)
                if len(rows) < self.settings.batch_size: break
            return total

def run_forever(settings: Settings):
    indexer=Indexer(settings)
    while True:
        try:
            indexed=indexer.run_once(settings.batch_size)
            if indexed == 0:
                time.sleep(1)
        except Exception:
            time.sleep(5)
            raise
