"""API-layer integration tests. Runs against the real lifespan (real index +
embedder load) but without requiring GEMINI_API_KEY/SARVAM_API_KEY -- these
tests exercise input validation and graceful-degradation paths, which don't
need live external calls."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_index(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0


def test_query_requires_text_or_audio(client):
    res = client.post("/api/query", data={})
    assert res.status_code == 400


def test_query_without_llm_key_returns_503(client):
    res = client.post("/api/query", data={"text": "what is a corporation"})
    assert res.status_code == 503
    assert "GEMINI_API_KEY" in res.json()["detail"]


def test_metrics_empty_state(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "count" in res.json()


def test_benchmark_without_llm_key_returns_503(client):
    res = client.post("/benchmark?n=1")
    assert res.status_code == 503
