// Display helpers. Missing values render as "n/a", never as zero.
import type { EvidenceClass, Experiment } from "./api";

export const NA = "n/a";

export function num(x: number | null | undefined, digits = 1, unit = ""): string {
  return x === null || x === undefined || !Number.isFinite(x) ? NA : `${x.toFixed(digits)}${unit}`;
}

export const speedup = (x: number | null | undefined) => num(x, 3, "×");
export const usd = (x: number | null | undefined) => (x === null || x === undefined ? "unknown" : `$${x.toFixed(2)}`);

export function configLabel(config: Record<string, number> | null): string {
  if (!config) return "target-only";
  return `lookahead ${config.lookahead_tokens} · p_min ${config.draft_p_min}`;
}

export const EVIDENCE: Record<EvidenceClass, { tag: string; meaning: string }> = {
  local: { tag: "LOCAL", meaning: "Measured on this machine. Not comparable to the CUDA flagship." },
  observed: { tag: "OBSERVED", meaning: "Imported from an externally scored run with a source ID." },
  replay: { tag: "REPLAY", meaning: "Recorded run re-served from the ledger. Not re-measured." },
  fixture: { tag: "FIXTURE", meaning: "Test data. Not a measurement." },
};

export const ACTION_LABEL: Record<string, string> = {
  run_baseline: "Baseline",
  propose_new: "New candidate",
  replicate: "Replicate",
  audit: "Audit",
  keep: "Keep",
  reject: "Reject",
  request_replicate: "Needs replication",
  flag_investigate: "Flag: unstable",
  promote: "Promote",
  stop: "Stop",
  investigate: "Investigate",
};

export type Tone = "good" | "bad" | "warn" | "busy" | "muted";

/** One plain-language status per run: the lifecycle state, refined by the controller's verdict. */
export function runStatus(e: Experiment): { text: string; tone: Tone } {
  if (e.state === "QUEUED") return { text: "Queued", tone: "busy" };
  if (e.state === "RUNNING") return { text: "Measuring", tone: "busy" };
  if (e.state === "FAILED") return { text: "Failed", tone: "bad" };
  if (e.state === "TIMED_OUT") return { text: "Timed out", tone: "bad" };
  if (e.state === "CANCELLED") return { text: "Cancelled", tone: "muted" };
  if (e.state === "INVALID") return { text: "Invalid", tone: "bad" };
  if (e.verdict === "promoted") return { text: "Promoted", tone: "good" };
  if (e.verdict === "reject") return { text: "Valid · rejected", tone: "muted" };
  if (e.verdict === "request_replicate") return { text: "Valid · unconfirmed", tone: "warn" };
  if (e.verdict === "flag_investigate") return { text: "Valid · unstable", tone: "warn" };
  if (e.verdict === "keep") return { text: e.split === "audit" ? "Audit passed" : "Valid · contender", tone: "good" };
  return { text: e.kind === "baseline" ? "Valid reference" : "Valid", tone: "good" };
}

export function tokenMatch(e: Experiment): string {
  const v = e.validation;
  if (!v.compared) return NA;
  return `${v.compared - (v.truncated_prefix ?? 0) - (v.diverged ?? 0)}/${v.compared}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return NA;
  return seconds < 90 ? `${seconds.toFixed(0)}s` : `${(seconds / 60).toFixed(1)}m`;
}

/** Depth in the hypothesis lineage (parent links), capped so the rail stays readable. */
export function lineageDepth(e: Experiment, byId: Map<string, Experiment>): number {
  let depth = 0;
  let cur: Experiment | undefined = e;
  while (cur?.parent_id && depth < 3) {
    cur = byId.get(cur.parent_id);
    depth += 1;
  }
  return depth;
}
