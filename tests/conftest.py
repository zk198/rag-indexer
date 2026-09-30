import os

os.environ.setdefault("RAG_POSTGRES_DSN", "postgresql://test:test@postgres.invalid/test")
os.environ.setdefault("RAG_QDRANT_URL", "http://qdrant.invalid:6333")
