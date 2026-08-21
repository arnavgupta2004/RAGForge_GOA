"""benchmark_retrieval.py uses a linear-interpolation percentile (matching
the organizer-shared reference script's convention) rather than
benchmark.py's nearest-rank percentile -- both are legitimate, but they can
disagree slightly, so each is tested against its own module directly."""

from benchmark_retrieval import percentile


def test_percentile_p50_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_percentile_p100_is_max():
    assert percentile([5, 1, 9, 3], 100) == 9


def test_percentile_p0_is_min():
    assert percentile([5, 1, 9, 3], 0) == 1


def test_percentile_single_value():
    assert percentile([42.0], 95) == 42.0
