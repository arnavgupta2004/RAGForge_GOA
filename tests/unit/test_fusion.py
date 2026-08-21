from src.retrieval.fusion import fuse


def test_fuse_dense_only_alpha_1():
    dense = {1: 0.9, 2: 0.5, 3: 0.1}
    result = fuse(dense, {}, alpha=1.0)
    assert result[0].chunk_idx == 1
    assert result[-1].chunk_idx == 3


def test_fuse_sparse_only_alpha_0():
    sparse = {1: 2.0, 2: 10.0, 3: 5.0}
    result = fuse({}, sparse, alpha=0.0)
    assert result[0].chunk_idx == 2


def test_fuse_combines_both_signals():
    dense = {1: 0.9, 2: 0.2}
    sparse = {1: 1.0, 2: 10.0}
    # heavy sparse weight should favor idx 2 despite low dense score
    result = fuse(dense, sparse, alpha=0.1)
    assert result[0].chunk_idx == 2


def test_fuse_handles_disjoint_candidate_sets():
    dense = {1: 0.8}
    sparse = {2: 5.0}
    result = fuse(dense, sparse, alpha=0.5)
    ids = {c.chunk_idx for c in result}
    assert ids == {1, 2}


def test_fuse_empty_inputs():
    assert fuse({}, {}, alpha=0.5) == []
