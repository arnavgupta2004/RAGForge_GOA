"""API-layer integration tests. Runs against the real lifespan (real index +
embedder load). The "missing key" tests simulate absence by clearing
api.main.STATE directly rather than depending on the ambient environment --
so they pass the same way whether or not GEMINI_API_KEY/SARVAM_API_KEY are
actually set. `test_query_with_text_returns_real_response` is the one test
that genuinely needs a working GEMINI_API_KEY and is skipped without one."""

import os

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
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
    original = main_module.STATE.get("llm_client")
    main_module.STATE["llm_client"] = None
    try:
        res = client.post("/api/query", data={"text": "what is a corporation"})
        assert res.status_code == 503
        assert "GEMINI_API_KEY" in res.json()["detail"]
    finally:
        main_module.STATE["llm_client"] = original


def test_metrics_empty_or_populated_state(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "count" in res.json()


def test_benchmark_without_llm_key_returns_503(client):
    original = main_module.STATE.get("llm_client")
    main_module.STATE["llm_client"] = None
    try:
        res = client.post("/benchmark?n=1")
        assert res.status_code == 503
    finally:
        main_module.STATE["llm_client"] = original


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="requires a real GEMINI_API_KEY")
def test_query_with_text_returns_real_response(client):
    res = client.post("/api/query", data={"text": "what is a corporation"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("answered", "refused")
    assert "total" in body["latency_ms"]
