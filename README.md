# RAG Indexer

Stateless indexing worker that projects authoritative document chunks from PostgreSQL into rebuildable dense and sparse vectors in Qdrant.

## Responsibilities
- Read chunk records from PostgreSQL with tenant and source metadata.
- Generate dense embeddings with FastEmbed (default model: `BAAI/bge-small-en-v1.5`).
- Generate sparse BM25 vectors using the Qdrant BM25 FastEmbed model.
- Create/maintain the Qdrant collection and payload indexes required for hybrid retrieval.
- Upsert deterministic chunk points with tenant/source/user metadata for downstream filtering.

## Data flow
`rag-ingestion → PostgreSQL chunks/outbox → rag-indexer → Qdrant dense+sparse vectors → rag-retrieval`.

PostgreSQL remains authoritative. Qdrant is derived, rebuildable index state.

## Configuration
PostgreSQL and Qdrant topology must be supplied explicitly by the deployment environment; do not assume a localhost endpoint inside containers. Collection and embedding-model settings are also environment-configured. See the settings module and `ai-infra` deployment environment for the exact current variable names.

## Development
Python 3.11+; use uv.
```bash
uv sync --extra test
uv run pytest
```
Run with the platform composition in `ai-infra`. PostgreSQL must contain chunk records and Qdrant must be reachable.

## Operations
Indexing is repeatable and Qdrant can be rebuilt from PostgreSQL. Preserve tenant metadata on every point. Monitor database lag, embedding latency, batch throughput, Qdrant upsert failures and model-cache storage. Dense/sparse collection topology must remain compatible with `rag-retrieval`.
