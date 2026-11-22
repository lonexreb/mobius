import { describe, expect, it } from "vitest";
import type { Experiment } from "./api";
import { lineageDepth, num, runStatus, speedup, tokenMatch, usd } from "./format";

const exp = (over: Partial<Experiment>): Experiment =>
  ({ id: "r", seq: 0, kind: "candidate", state: "VALID", verdict: null, split: "dev", parent_id: null,
     validation: {}, metrics: {}, ...over }) as Experiment;

describe("format", () => {
  it("never renders a missing value as zero", () => {
    expect(num(null)).toBe("n/a");
    expect(num(undefined)).toBe("n/a");
    expect(num(Number.NaN)).toBe("n/a");
    expect(speedup(undefined)).toBe("n/a");
    expect(usd(null)).toBe("unknown");
    expect(usd(0)).toBe("$0.00");
  });

  it("an invalid run can never read as good, whatever its verdict", () => {
    expect(runStatus(exp({ state: "INVALID", verdict: "keep" })).tone).toBe("bad");
    expect(runStatus(exp({ state: "FAILED" })).tone).toBe("bad");
    expect(runStatus(exp({ verdict: "promoted" }))).toEqual({ text: "Promoted", tone: "good" });
    expect(runStatus(exp({ verdict: "request_replicate" })).tone).toBe("warn");
  });

  it("counts matched prompts from truncated and diverged", () => {
    expect(tokenMatch(exp({ validation: { compared: 24, truncated_prefix: 12, diverged: 4 } }))).toBe("8/24");
    expect(tokenMatch(exp({}))).toBe("n/a");
  });

  it("computes lineage depth from parent links", () => {
    const a = exp({ id: "a" }), b = exp({ id: "b", parent_id: "a" }), c = exp({ id: "c", parent_id: "b" });
    const byId = new Map([a, b, c].map((e) => [e.id, e]));
    expect([a, b, c].map((e) => lineageDepth(e, byId))).toEqual([0, 1, 2]);
  });
});
