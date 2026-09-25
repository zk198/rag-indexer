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

    @classmethod
    def from_env(cls):
        return cls(
            os.environ["RAG_POSTGRES_DSN"],
            os.getenv("RAG_QDRANT_URL", "http://qdrant:6333"),
            os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks"),
            os.getenv("RAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5"),
            os.getenv("RAG_SPARSE_MODEL", "Qdrant/bm25"),
            int(os.getenv("RAG_INDEX_BATCH_SIZE", "64")),
        )
