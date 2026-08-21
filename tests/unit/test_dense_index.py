import numpy as np

from src.retrieval.dense_index import build_dense_index, search_dense


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_search_returns_exact_nearest_neighbor():
    vecs = _normalize(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32"))
    index = build_dense_index(vecs)
    query = _normalize(np.array([1.0, 0.05], dtype="float32"))
    scores, ids = search_dense(index, query, top_k=2)
    assert ids[0] == 0
    assert scores[0] > scores[1]


def test_search_top_k_capped_at_corpus_size():
    vecs = _normalize(np.random.rand(3, 8).astype("float32"))
    index = build_dense_index(vecs)
    scores, ids = search_dense(index, vecs[0], top_k=10)
    assert len(ids) == 3
