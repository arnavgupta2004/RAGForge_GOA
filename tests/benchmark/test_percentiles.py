"""Unit tests for the benchmark harness's pure percentile math -- these don't
need a live API key, unlike the harness's actual run() which makes real
generation calls."""

from benchmark import percentile, summarize


def test_percentile_p50_of_sorted_range():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50 or percentile(values, 50) == 51


def test_percentile_p100_is_max():
    values = [5, 1, 9, 3]
    assert percentile(values, 100) == 9


def test_percentile_p0_is_min_ish():
    values = [5, 1, 9, 3]
    assert percentile(values, 0) == 1


def test_percentile_empty_list():
    assert percentile([], 50) == 0.0


def test_summarize_contains_all_expected_keys():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = summarize(values)
    for key in ("p50", "p70", "p90", "p95", "p99", "p100", "mean", "min", "max", "n"):
        assert key in result
    assert result["n"] == 5
    assert result["min"] == 10.0
    assert result["max"] == 50.0
    assert result["mean"] == 30.0


def test_summarize_empty_returns_empty_dict():
    assert summarize([]) == {}
