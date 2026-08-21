from src.retrieval.confidence import calibrated_confidence
from src.retrieval.models import RetrievedChunk
from src.chunking.models import Chunk


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="c1", doc_id="d1", parent_id="d1", source="test",
        text="text", original_text="text", chunking_strategy="atomic",
        position=0, token_count=2, query_id=1, query_type="DESCRIPTION", is_selected=True,
    )


def test_confidence_none_returns_zero():
    assert calibrated_confidence(None) == 0.0


def test_confidence_dense_only_is_passthrough():
    c = RetrievedChunk(chunk=make_chunk(), dense_score=0.73, final_score=0.73)
    assert calibrated_confidence(c) == 0.73


def test_confidence_sparse_saturates_into_0_1():
    c = RetrievedChunk(chunk=make_chunk(), sparse_score=100.0, final_score=1.0)
    conf = calibrated_confidence(c)
    assert 0.0 < conf < 1.0


def test_confidence_takes_max_of_available_signals():
    c = RetrievedChunk(chunk=make_chunk(), dense_score=0.2, sparse_score=50.0, final_score=1.0)
    conf = calibrated_confidence(c)
    assert conf > 0.2  # sparse signal should dominate here
