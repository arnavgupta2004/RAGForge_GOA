import type { HealthResponse, MetricsResponse, PipelineResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8420";

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
  const res = await fetch(`${API_BASE}/api/query`, { method: "POST", body: form });
  return handle<PipelineResponse>(res);
}

export async function queryAudio(blob: Blob): Promise<PipelineResponse> {
  const form = new FormData();
  form.append("audio", blob, "audio.webm");
  const res = await fetch(`${API_BASE}/api/query`, { method: "POST", body: form });
  return handle<PipelineResponse>(res);
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return handle<HealthResponse>(res);
}

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`);
  return handle<MetricsResponse>(res);
}

export async function runBenchmark(n = 20): Promise<unknown> {
  const res = await fetch(`${API_BASE}/benchmark?n=${n}`, { method: "POST" });
  return handle(res);
}
