from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    postgres_dsn: str
    qdrant_url: str
    collection: str
    dense_model: str
    sparse_model: str
    batch_size: int = 64
    outbox_lock_timeout_seconds: int = 300
    outbox_retry_base_seconds: int = 5
    outbox_retry_max_seconds: int = 300

    @classmethod
    def from_env(cls):
        return cls(
            os.environ["RAG_POSTGRES_DSN"],
            os.getenv("RAG_QDRANT_URL", "http://qdrant:6333"),
            os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks"),
            os.getenv("RAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5"),
            os.getenv("RAG_SPARSE_MODEL", "Qdrant/bm25"),
            int(os.getenv("RAG_INDEX_BATCH_SIZE", "64")),
            int(os.getenv("RAG_OUTBOX_LOCK_TIMEOUT_SECONDS", "300")),
            int(os.getenv("RAG_OUTBOX_RETRY_BASE_SECONDS", "5")),
            int(os.getenv("RAG_OUTBOX_RETRY_MAX_SECONDS", "300")),
        )
