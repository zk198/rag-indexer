# RAG Indexer

Consumes PostgreSQL chunk data and produces rebuildable dense + sparse vectors in Qdrant.

Dense embeddings use FastEmbed and sparse embeddings use the Qdrant BM25 model. Qdrant supports hybrid retrieval with dense and sparse vectors; retrieval and tenant filtering are implemented in the downstream retrieval service.

The indexer is deliberately stateless apart from model caches. PostgreSQL remains authoritative.
