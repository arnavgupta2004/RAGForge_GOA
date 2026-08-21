from src.retrieval.query_variants import generate_variants


def test_strips_interrogative_scaffold():
    variants = generate_variants("what is a corporation?")
    assert "a corporation" in [v.lower() for v in variants]


def test_original_always_included():
    variants = generate_variants("carbide")
    assert "carbide" in variants


def test_no_duplicate_variants():
    variants = generate_variants("who is who")
    assert len(variants) == len(set(v.lower() for v in variants))


def test_respects_max_variants():
    variants = generate_variants("what is the definition of gross wages", max_variants=1)
    assert len(variants) == 1
