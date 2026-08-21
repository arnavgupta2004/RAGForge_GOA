export type Status = "answered" | "refused" | "error";

export interface SourceOut {
  chunk_id: string;
  doc_id: string;
  snippet: string;
  chunking_strategy: string;
  expanded: boolean;
  dense_score: number | null;
  sparse_score: number | null;
  fused_score: number | null;
  rerank_score: number | null;
  final_score: number;
}

export interface PipelineResponse {
  request_id: string;
  status: Status;
  answer: string;
  sources: SourceOut[];
  confidence: number;
  retrieval_strategy: string | null;
  retrieval_reason: string | null;
  query_variants: string[];
  transcript: string | null;
  detected_language: string | null;
  reason: string | null;
  prompt_injection_detected: boolean;
  groundedness_overlap: number | null;
  latency_ms: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  uptime_s: number;
  chunks_indexed: number;
  llm_configured: boolean;
  asr_configured: boolean;
}

export interface MetricsResponse {
  count: number;
  message?: string;
  latency_ms?: {
    p50: number;
    p70: number;
    p90: number;
    p95: number;
    p99: number;
    p100: number;
    mean: number;
    min: number;
    max: number;
  };
  status_counts?: Record<string, number>;
  strategy_counts?: Record<string, number>;
}

export type PipelineStage =
  | "idle"
  | "listening"
  | "asr"
  | "router"
  | "retrieval"
  | "context"
  | "generation"
  | "guardrail"
  | "done";
