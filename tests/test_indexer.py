from rag_indexer.indexer import Indexer

def test_indexer_exposes_index_rows():
    assert hasattr(Indexer, "index_rows")
