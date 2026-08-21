from src.chunking.pipeline import PassageRecord, chunk_passage_adaptive, chunk_passage_baseline
from src.chunking.strategies import count_tokens, fixed_token_chunks, sentence_aware_chunks, split_sentences

SHORT_TEXT = "A corporation is a company recognized as a legal entity."
LONG_TEXT = (
    "A company is incorporated in a specific nation, often within the bounds of a smaller "
    "subset of that nation, such as a state or province. The corporation is then governed by "
    "the laws of incorporation in that state. A corporation may issue stock, either private or "
    "public, or may be classified as a non-stock corporation. If stock is issued, the corporation "
    "will usually be governed by its shareholders, either directly or indirectly. Corporations are "
    "owned by their stockholders who share in profits and losses generated through operations."
)


def test_short_passage_stays_atomic():
    passage = PassageRecord("q1_p0", SHORT_TEXT, query_id=1, query_type="DESCRIPTION", is_selected=True)
    chunks = chunk_passage_adaptive(
        passage, atomic_max_tokens=90, fixed_window=64, sentence_per_chunk=2, overlap_window=64, overlap_tokens=16
    )
    assert len(chunks) == 1
    assert chunks[0].chunking_strategy == "atomic"
    assert chunks[0].parent_id == "q1_p0"
    assert chunks[0].text == SHORT_TEXT


def test_long_passage_splits_three_ways():
    passage = PassageRecord("q2_p0", LONG_TEXT, query_id=2, query_type="DESCRIPTION", is_selected=False)
    chunks = chunk_passage_adaptive(
        passage, atomic_max_tokens=30, fixed_window=20, sentence_per_chunk=1, overlap_window=20, overlap_tokens=5
    )
    strategies = {c.chunking_strategy for c in chunks}
    assert strategies == {"fixed", "sentence", "overlapping"}
    assert all(c.parent_id == "q2_p0" for c in chunks)
    assert all(c.original_text == LONG_TEXT for c in chunks)


def test_neighbor_linkage():
    passage = PassageRecord("q3_p0", LONG_TEXT, query_id=3, query_type="DESCRIPTION", is_selected=False)
    chunks = chunk_passage_adaptive(
        passage, atomic_max_tokens=30, fixed_window=20, sentence_per_chunk=1, overlap_window=20, overlap_tokens=5
    )
    fixed_chunks = [c for c in chunks if c.chunking_strategy == "fixed"]
    assert len(fixed_chunks) > 1
    # middle chunk should link to both neighbors
    middle = fixed_chunks[1]
    assert fixed_chunks[0].chunk_id in middle.neighbor_chunk_ids
    assert fixed_chunks[2].chunk_id in middle.neighbor_chunk_ids


def test_baseline_ignores_length_uniformly():
    short_passage = PassageRecord("q4_p0", SHORT_TEXT, query_id=4, query_type="DESCRIPTION", is_selected=True)
    chunks = chunk_passage_baseline(short_passage, window_tokens=64)
    assert all(c.chunking_strategy == "baseline_fixed" for c in chunks)


def test_fixed_token_chunks_no_overlap_covers_all_tokens():
    spans = fixed_token_chunks(LONG_TEXT, window_tokens=20, overlap_tokens=0)
    total_tokens = sum(count_tokens(s) for s in spans)
    assert total_tokens == count_tokens(LONG_TEXT)


def test_fixed_token_chunks_overlap_produces_more_tokens_than_source():
    no_overlap = fixed_token_chunks(LONG_TEXT, window_tokens=20, overlap_tokens=0)
    with_overlap = fixed_token_chunks(LONG_TEXT, window_tokens=20, overlap_tokens=5)
    assert len(with_overlap) >= len(no_overlap)


def test_sentence_split_reasonable():
    sentences = split_sentences(LONG_TEXT)
    assert len(sentences) >= 4
    assert all(s.strip() for s in sentences)


def test_sentence_aware_chunks_short_text_passthrough():
    chunks = sentence_aware_chunks(SHORT_TEXT, sentences_per_chunk=5)
    assert chunks == [SHORT_TEXT]
