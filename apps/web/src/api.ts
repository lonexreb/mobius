// Typed client for the Möbius Lab API (v1). Mirrors apps/api/src/mobius_lab/schemas.py.

export type Mode = "local" | "modal" | "autolab" | "replay";
export type EvidenceClass = "local" | "observed" | "replay" | "fixture";

export interface Budget {
  approved_spend_usd: number;
  budget_approved: boolean;
  reserved_usd: number;
  observed_spend_usd: number;
  unknown_cost_events: number;
  by_category: Record<string, { actual_usd: number; unknown_events: number }>;
  max_trials: number;
  trials_used: number;
  max_wall_seconds: number;
  wall_seconds_used: number;
}

export interface Decision {
  id: number;
  experiment_id: string | null;
  action: string;
  rationale: string;
  evidence_ids: string[];
  uncertainty: string | null;
  controller_version: string;
  created_at: string;
}

export interface Metrics {
  throughput_tokens_per_sec?: number;
  throughput_by_domain?: Record<string, number>;
  control_throughput_tokens_per_sec?: number;
  p50_latency_ms?: number;
  p95_latency_ms?: number;
  time_to_first_token_ms?: number | null;
  draft_acceptance_rate?: number | null;
  speedup_vs_target?: number;
  speedup_by_domain?: Record<string, number>;
  speedup_ci_low?: number;
  speedup_ci_high?: number;
  speedup_ci_method?: string;
  total_output_tokens?: number;
  total_prompt_tokens?: number;
  prompts?: number;
  server_rss_gb?: number | null;
  wall_seconds?: number;
  warmup_seconds?: number;
}

export interface Validation {
  gates_passed?: string[];
  gates_failed?: string[];
  notes?: Record<string, string>;
  exact_token_match?: boolean | null;
  truncated_prefix?: number;
  diverged?: number;
  compared?: number | null;
  target_self_matched?: number | null;
  target_self_compared?: number | null;
  control_matched_reference?: number | null;
}

export interface Experiment {
  id: string;
  study_id: string;
  seq: number;
  policy_arm: string;
  kind: "baseline" | "candidate";
  action: string;
  config: Record<string, number> | null;
  split: string;
  hypothesis: string | null;
  rationale: string | null;
  parent_id: string | null;
  state: string;
  verdict: string | null;
  evidence_class: EvidenceClass;
  source: Record<string, string | null>;
  repro: Record<string, unknown>;
  metrics: Metrics;
  validation: Validation;
  failure: string | null;
  wall_seconds: number | null;
  created_at: string;
  finished_at: string | null;
}

export interface RequestRow {
  prompt_id: string;
  domain: string;
  decode_tps: number;
  e2e_ms: number;
  ttft_ms: number | null;
  output_tokens: number;
  prompt_tokens: number;
  draft_n: number | null;
  draft_n_accepted: number | null;
}

export interface ExperimentDetail extends Experiment {
  requests: RequestRow[] | null;
  control_requests: RequestRow[] | null;
  decisions: Decision[];
}

export interface Study {
  id: string;
  goal: string;
  template_id: string;
  mode: Mode;
  policy_arm: string;
  state: string;
  frozen_eval_hash: string | null;
  replay_of: string | null;
  notes: string | null;
  created_at: string;
  budget: Budget;
  evidence_label: string;
  best_valid: Experiment | null;
  promoted: Experiment | null;
  latest_decision: Decision | null;
  failure_reason: string | null;
}

export interface Template {
  id: string;
  title: string;
  runnable_modes: Mode[];
  label: string;
  target_model: string | { repo: string; revision: string };
  draft_model: string | { repo: string; revision: string };
  engine: string;
  primary_metric: string;
  validity: string;
  search_space: Record<string, number[]>;
  estimated_max_usd_per_run?: number;
  pricing_source?: string;
  sources: { domain: string; repo: string; revision: string; license: string; selected: number; url: string }[];
  splits: Record<string, { sha256: string; count: number; role: string }>;
}

export interface Connection {
  ready: boolean;
  blocker: string | null;
  detail?: string;
}
export type Connections = Record<"local" | "mobius_cli" | "autolab" | "gpu" | "egbe" | "llm_proposer", Connection>;

export interface StudyCreate {
  template_id: string;
  goal: string;
  mode: Mode;
  policy_arm: string;
  budget: { max_trials: number; approved_spend_usd: number; max_wall_seconds: number };
  replay_of?: string;
  notes?: string;
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public blockers: string[] = []) {
    super(message);
  }
}

const BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, headers: { "Content-Type": "application/json" } });
  } catch {
    throw new ApiError("Cannot reach the Möbius Lab API on 127.0.0.1:8000. Is it running?", 0);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    if (Array.isArray(detail)) {
      throw new ApiError(detail.map((d) => `${d.loc?.slice(1).join(".")}: ${d.msg}`).join("; "), res.status);
    }
    if (detail && typeof detail === "object") throw new ApiError(detail.message, res.status, detail.blockers ?? []);
    throw new ApiError(typeof detail === "string" ? detail : `Request failed (${res.status})`, res.status);
  }
  return (path.endsWith("/report") ? res.text() : res.json()) as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  templates: () => request<{ templates: Template[]; policy_arms: string[] }>("/templates"),
  connections: () => request<Connections>("/connections"),
  studies: () => request<Study[]>("/studies"),
  study: (id: string) => request<Study>(`/studies/${id}`),
  experiments: (id: string) => request<Experiment[]>(`/studies/${id}/experiments`),
  decisions: (id: string) => request<Decision[]>(`/studies/${id}/decisions`),
  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${id}`),
  create: (body: StudyCreate) => post<Study>("/studies", body),
  act: (id: string, verb: "start" | "resume" | "pause" | "cancel" | "approve-budget") =>
    post<Study>(`/studies/${id}/${verb}`),
  report: (id: string) => request<string>(`/studies/${id}/report`),
  exportJson: (id: string) => request<unknown>(`/studies/${id}/export.json`),
  eventsUrl: (id: string, after: number) => `${BASE}/studies/${id}/events?after_event_id=${after}`,
};
