import { useEffect, useRef, useState } from "react";
import { api, type ExperimentDetail } from "../api";
import { ACTION_LABEL, configLabel, duration, EVIDENCE, num, runStatus, speedup, tokenMatch } from "../format";

interface Props {
  experimentId: string;
  /** Changes whenever the run's state or verdict changes, so open evidence never goes stale. */
  version: string;
  onClose: () => void;
  onSelect: (id: string) => void;
}

const short = (s: unknown, n = 12) => (typeof s === "string" && s ? `${s.slice(0, n)}…` : "not available");

export function EvidenceDrawer({ experimentId, version, onClose, onSelect }: Props) {
  const [exp, setExp] = useState<ExperimentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setExp(null);
  }, [experimentId]);
  useEffect(() => {
    let stale = false;
    setError(null);
    api.experiment(experimentId).then((e) => !stale && setExp(e)).catch((e) => !stale && setError(e.message));
    return () => { stale = true; };
  }, [experimentId, version]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const m = exp?.metrics, v = exp?.validation, r = (exp?.repro ?? {}) as Record<string, unknown>;
  const hw = (r.hardware ?? {}) as Record<string, unknown>;
  const domains = m?.speedup_by_domain ? Object.entries(m.speedup_by_domain) : [];
  const maxDomain = Math.max(1.2, ...domains.map(([, x]) => x));

  return (
    <aside className="drawer" role="dialog" aria-modal="false" aria-label={`Evidence for ${experimentId}`}>
      <header>
        <div>
          <p className="eyebrow">Evidence</p>
          <h2>{experimentId}</h2>
        </div>
        <button ref={closeRef} className="btn ghost" onClick={onClose} aria-label="Close evidence">Close ✕</button>
      </header>

      {error && <p className="notice bad" role="alert">{error}</p>}
      {!exp && !error && <p className="dim">Loading evidence…</p>}

      {exp && m && v && (
        <>
          <p className="badges">
            <span className={`pill tone-${runStatus(exp).tone}`}>{runStatus(exp).text}</span>
            <span className="tag" title={EVIDENCE[exp.evidence_class].meaning}>{EVIDENCE[exp.evidence_class].tag}</span>
            <span className="tag">{exp.split} split</span>
            <span className="tag">{ACTION_LABEL[exp.action] ?? exp.action}</span>
          </p>
          <p className="dim">{EVIDENCE[exp.evidence_class].meaning}</p>

          <section>
            <h3>Hypothesis</h3>
            <p>{exp.hypothesis ?? "None recorded."}</p>
            <p className="dim">Why this run: {exp.rationale}</p>
            <p><code>{configLabel(exp.config)}</code>{exp.parent_id && (
              <> · branched from <button className="link" onClick={() => onSelect(exp.parent_id!)}>{exp.parent_id}</button></>
            )}</p>
          </section>

          {exp.failure && <p className="notice bad">{exp.failure}</p>}

          <section>
            <h3>Validity gates</h3>
            <ul className="gates">
              {(v.gates_passed ?? []).map((g) => <li key={g} className="pass"><span aria-hidden="true">✓</span> {g} <em>passed</em></li>)}
              {(v.gates_failed ?? []).map((g) => (
                <li key={g} className="fail"><span aria-hidden="true">✕</span> {g} <em>failed</em>{v.notes?.[g] && <small>{v.notes[g]}</small>}</li>
              ))}
              {!v.gates_passed?.length && !v.gates_failed?.length && <li className="dim">No gates evaluated: the run produced no artifact.</li>}
            </ul>
            {exp.kind === "candidate" && v.compared ? (
              <p className="dim">
                Tokens matched on {tokenMatch(exp)} prompts. {v.truncated_prefix} stopped early with an identical prefix,{" "}
                {v.diverged} diverged mid-sequence. Any mismatch fails the lossless gate.
                {v.target_self_compared != null && <> Compared against the in-run target-only control; the target repeated
                  itself on {v.target_self_matched}/{v.target_self_compared} prompts in this run.</>}
              </p>
            ) : null}
          </section>

          <section>
            <h3>Measured metrics</h3>
            <dl className="metrics">
              {exp.kind === "candidate" && <><dt>Speedup vs target</dt><dd>{speedup(m.speedup_vs_target)}</dd>
                <dt>95% CI</dt><dd>{speedup(m.speedup_ci_low)} – {speedup(m.speedup_ci_high)}</dd>
                <dt>In-run control</dt><dd>{num(m.control_throughput_tokens_per_sec, 1, " tok/s")}</dd></>}
              <dt>Decode throughput</dt><dd>{num(m.throughput_tokens_per_sec, 1, " tok/s")}</dd>
              <dt>Latency p50 / p95</dt><dd>{num(m.p50_latency_ms, 0)} / {num(m.p95_latency_ms, 0, " ms")}</dd>
              <dt>Time to first token</dt><dd>{num(m.time_to_first_token_ms, 0, " ms")}</dd>
              <dt>Draft acceptance</dt><dd>{m.draft_acceptance_rate == null ? "n/a" : `${(m.draft_acceptance_rate * 100).toFixed(0)}%`}</dd>
              <dt>Output / prompt tokens</dt><dd>{num(m.total_output_tokens, 0)} / {num(m.total_prompt_tokens, 0)}</dd>
              <dt>Server RSS</dt><dd>{num(m.server_rss_gb, 1, " GB")}</dd>
              <dt>Wall / warmup</dt><dd>{duration(m.wall_seconds)} / {duration(m.warmup_seconds)}</dd>
              <dt>Billed cost</dt><dd>{exp.evidence_class === "observed" ? "see budget" : "$0.00 (no billed service)"}</dd>
            </dl>
            {m.speedup_ci_method && <p className="dim">CI: {m.speedup_ci_method}.</p>}
          </section>

          {domains.length > 0 && (
            <section>
              <h3>Speedup by domain</h3>
              <ul className="bars">
                {domains.map(([d, x]) => (
                  <li key={d}>
                    <span>{d}</span>
                    <span className="bar-track">
                      <span className="bar-ref" style={{ left: `${(1 / maxDomain) * 100}%` }} />
                      <span className="bar-fill" style={{ transform: `scaleX(${x / maxDomain})` }} />
                    </span>
                    <span className="mono">{speedup(x)}</span>
                  </li>
                ))}
              </ul>
              <p className="dim">Vertical rule marks 1.00× (target-only).</p>
            </section>
          )}

          <section>
            <h3>Decisions citing this run</h3>
            {exp.decisions.length === 0 ? <p className="dim">None yet.</p> : (
              <ol className="trace">
                {exp.decisions.map((d) => (
                  <li key={d.id}><strong>{ACTION_LABEL[d.action] ?? d.action}</strong> {d.rationale}
                    {d.uncertainty && <small>Uncertainty: {d.uncertainty}</small>}</li>
                ))}
              </ol>
            )}
          </section>

          <section>
            <h3>Provenance</h3>
            <dl className="metrics">
              <dt>Code</dt><dd>{short(r.git_sha)}{r.git_dirty ? " (dirty)" : ""}</dd>
              <dt>Evaluator sha256</dt><dd>{short(r.evaluator_sha256)}</dd>
              <dt>Prompt split sha256</dt><dd>{short(r.dataset_split_sha256)}</dd>
              <dt>Target model</dt><dd>{String(r.target_model ?? "n/a")} · {short(r.target_model_sha256)}</dd>
              <dt>Draft model</dt><dd>{r.draft_model ? `${String(r.draft_model)} · ${short(r.draft_model_sha256)}` : "n/a"}</dd>
              <dt>Engine</dt><dd>{String(r.engine_version ?? "not available")}</dd>
              <dt>Hardware</dt><dd>{String(hw.cpu ?? "not available")}{hw.accelerator ? ` · ${String(hw.accelerator)}` : ""}</dd>
              <dt>Source</dt><dd>{exp.source.run_id ? `${exp.source.provider} ${exp.source.run_id}` : "this machine"}</dd>
            </dl>
            {exp.source.run_url && <p><a href={exp.source.run_url} target="_blank" rel="noreferrer">Open source run</a></p>}
          </section>

          {exp.requests && exp.requests.length > 0 && (
            <section>
              <h3>Per-prompt measurements</h3>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Prompt</th><th>Domain</th><th>tok/s</th>{exp.control_requests && <th>Control</th>}<th>Out</th><th>Draft acc.</th></tr></thead>
                  <tbody>
                    {exp.requests.map((q, i) => (
                      <tr key={q.prompt_id}>
                        <td title={q.prompt_id}>{q.prompt_id.slice(0, 22)}</td><td>{q.domain}</td>
                        <td>{num(q.decode_tps, 1)}</td>
                        {exp.control_requests && <td>{num(exp.control_requests[i]?.decode_tps, 1)}</td>}
                        <td>{q.output_tokens}</td>
                        <td>{q.draft_n ? `${q.draft_n_accepted}/${q.draft_n}` : "n/a"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </aside>
  );
}
