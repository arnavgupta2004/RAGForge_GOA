"""Chunking strategy implementations.

All strategies operate on a single passage's plain text and return a list of
(text, token_count) spans in order. Token counts use tiktoken's cl100k_base
encoding as a fast, model-agnostic proxy for length -- we don't need exact
target-model tokenization, just a consistent unit for windowing and reporting.
"""

from __future__ import annotations

import pysbd
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def split_sentences(text: str) -> list[str]:
    sentences = _SEGMENTER.segment(text)
    return [s.strip() for s in sentences if s.strip()]


def fixed_token_chunks(text: str, window_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Sliding window over token ids, decoded back to text. overlap_tokens=0 is
    the naive non-overlapping baseline; >0 gives the overlapping-semantic-chunk
    strategy."""
    ids = _ENC.encode(text)
    if len(ids) <= window_tokens:
        return [text]
    step = window_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("overlap_tokens must be smaller than window_tokens")
    spans = []
    start = 0
    while start < len(ids):
        window = ids[start : start + window_tokens]
        spans.append(_ENC.decode(window))
        if start + window_tokens >= len(ids):
            break
        start += step
    return spans


def sentence_aware_chunks(text: str, sentences_per_chunk: int) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= sentences_per_chunk:
        return [text]
    return [
        " ".join(sentences[i : i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]
