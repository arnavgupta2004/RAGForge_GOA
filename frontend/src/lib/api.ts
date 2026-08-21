import type { HealthResponse, MetricsResponse, PipelineResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8420";

// Real measured latency is ~1-2s (see docs/latency.md); this is a generous
// upper bound so a hung connection or backend outage fails within a bounded
// time instead of leaving the UI waiting indefinitely -- not a target to hit.
const REQUEST_TIMEOUT_MS = 45_000;

async function withTimeout(input: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Request timed out. The backend may be unavailable -- please try again.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function queryText(text: string): Promise<PipelineResponse> {
  const form = new FormData();
  form.append("text", text);
  const res = await withTimeout(`${API_BASE}/api/query`, { method: "POST", body: form });
  return handle<PipelineResponse>(res);
}

export async function queryAudio(blob: Blob): Promise<PipelineResponse> {
  const form = new FormData();
  form.append("audio", blob, "audio.webm");
  const res = await withTimeout(`${API_BASE}/api/query`, { method: "POST", body: form });
  return handle<PipelineResponse>(res);
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await withTimeout(`${API_BASE}/health`);
  return handle<HealthResponse>(res);
}

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await withTimeout(`${API_BASE}/metrics`);
  return handle<MetricsResponse>(res);
}

export async function runBenchmark(n = 20): Promise<unknown> {
  const res = await withTimeout(`${API_BASE}/benchmark?n=${n}`, { method: "POST" });
  return handle(res);
}
