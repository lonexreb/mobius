import { useEffect, useState } from "react";
import { api, ApiError, type Budget, type Decision, type Experiment } from "../api";
import { ACTION_LABEL, configLabel, duration, EVIDENCE, lineageDepth, runStatus, speedup, tokenMatch, usd } from "../format";
import { useStudy } from "../useStudy";
import { EvidenceDrawer } from "./EvidenceDrawer";
import { SpeedupChart } from "./SpeedupChart";

type Verb = "start" | "resume" | "pause" | "cancel" | "approve-budget";

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

function Elapsed({ since }: { since: string }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(t);
  }, []);
  return <>{Math.max(0, Math.round((Date.now() - Date.parse(since)) / 1000))}s</>;
}

function Timeline({ experiments, selected, onSelect }: { experiments: Experiment[]; selected: string | null; onSelect: (id: string) => void }) {
  const byId = new Map(experiments.map((e) => [e.id, e]));
  if (experiments.length === 0) {
    return <div className="empty"><p>No runs yet.</p><p className="dim">The first run is always the target-only baseline.</p></div>;
  }
  return (
    <ol className="timeline">
      {experiments.map((e) => {
        const status = runStatus(e), busy = status.tone === "busy";
        return (
          <li key={e.id} style={{ "--depth": lineageDepth(e, byId) } as React.CSSProperties} className={e.parent_id ? "has-parent" : ""}>
            <button className={`node ${selected === e.id ? "is-selected" : ""}`} onClick={() => onSelect(e.id)} aria-pressed={selected === e.id}>
              <span className={`dot tone-${status.tone} ${busy ? "pulse" : ""}`} aria-hidden="true" />
              <span className="node-main">
                <span className="node-title">
                  <span className="seq">#{e.seq}</span> {ACTION_LABEL[e.action] ?? e.action}
                  {e.split === "audit" && <span className="tag">audit split</span>}
                </span>
                <span className="node-config mono">{configLabel(e.config)}</span>
                {e.parent_id && <span className="node-parent">↳ from #{byId.get(e.parent_id)?.seq ?? "?"}</span>}
              </span>
              <span className="node-side">
                <span className={`pill tone-${status.tone}`}>{status.text}{busy && e.state === "RUNNING" && <> · <Elapsed since={e.created_at} /></>}</span>
                <span className="mono">
                  {e.kind === "candidate" && e.metrics.speedup_vs_target !== undefined
                    ? `${speedup(e.metrics.speedup_vs_target)} · ${tokenMatch(e)} matched`
                    : e.metrics.throughput_tokens_per_sec !== undefined ? `${e.metrics.throughput_tokens_per_sec.toFixed(1)} tok/s` : ""}
                </span>
                <span className="tag" title={EVIDENCE[e.evidence_class].meaning}>{EVIDENCE[e.evidence_class].tag}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function DecisionPanel({ decisions, state, onSelect }: { decisions: Decision[]; state: string; onSelect: (id: string) => void }) {
  const heading = state === "RUNNING" ? "Current step"
    : state === "COMPLETED" ? "Final decision"
    : `Latest decision (study ${state.toLowerCase()}, not final)`;
  const latest = decisions[decisions.length - 1];
  if (!latest) {
    return <div className="empty"><p>No decision yet.</p><p className="dim">Start the study: Möbius first measures the target-only baseline, because nothing can be compared without it.</p></div>;
  }
  return (
    <>
      <p className="eyebrow">{heading}</p>
      <p className="decision-action">{ACTION_LABEL[latest.action] ?? latest.action}</p>
      <p>{latest.rationale}</p>
      {latest.uncertainty && <p className="notice warn"><strong>Uncertainty:</strong> {latest.uncertainty}</p>}
      {latest.evidence_ids.length > 0 && (
        <p className="cites">Evidence: {latest.evidence_ids.map((id) => <button key={id} className="link mono" onClick={() => onSelect(id)}>{id}</button>)}</p>
      )}
      <details>
        <summary>Full decision trace ({decisions.length})</summary>
        <ol className="trace">
          {decisions.map((d) => (
            <li key={d.id}>
              <strong>{ACTION_LABEL[d.action] ?? d.action}</strong>
              {d.experiment_id && <> · <button className="link mono" onClick={() => onSelect(d.experiment_id!)}>{d.experiment_id}</button></>}
              <span>{d.rationale}</span>
            </li>
          ))}
        </ol>
      </details>
      <p className="dim">Controller {latest.controller_version}: transparent rules, not a learned or LLM policy.</p>
    </>
  );
}

function BudgetGauge({ b }: { b: Budget }) {
  const pct = (n: number, d: number) => `scaleX(${d > 0 ? Math.min(1, n / d) : 0})`;
  const cats = Object.entries(b.by_category);
  return (
    <>
      <div className="gauge"><div className="gauge-row"><span>Scored runs</span><span className="mono">{b.trials_used} / {b.max_trials}</span></div>
        <div className="gauge-track"><div className="gauge-fill" style={{ transform: pct(b.trials_used, b.max_trials) }} /></div></div>
      <div className="gauge"><div className="gauge-row"><span>Wall time</span><span className="mono">{duration(b.wall_seconds_used)} / {duration(b.max_wall_seconds)}</span></div>
        <div className="gauge-track"><div className="gauge-fill" style={{ transform: pct(b.wall_seconds_used, b.max_wall_seconds) }} /></div></div>
      <dl className="metrics">
        <dt>Approved cap</dt><dd>{usd(b.approved_spend_usd)}{b.approved_spend_usd > 0 && !b.budget_approved ? " (not yet approved)" : ""}</dd>
        <dt>Reserved</dt><dd>{usd(b.reserved_usd)}</dd>
        <dt>Observed spend</dt><dd>{usd(b.observed_spend_usd)}</dd>
        <dt>Unknown-cost events</dt><dd>{b.unknown_cost_events}</dd>
        {cats.map(([name, c]) => (
          <div key={name} className="sub"><dt>{name.replace(/_/g, " ")}</dt><dd>{usd(c.actual_usd)}{c.unknown_events ? ` + ${c.unknown_events} unknown` : ""}</dd></div>
        ))}
      </dl>
      <p className="dim">Hugging Face, Egbe, storage and hosting: not used, nothing billed. Local electricity is not metered (unknown, not zero).</p>
    </>
  );
}

export function StudyPage({ id }: { id: string }) {
  const { study, experiments, decisions, error, live, refresh } = useStudy(id);
  const [selected, setSelected] = useState<string | null>(null);
  const [actionError, setActionError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState<Verb | null>(null);
  const [copied, setCopied] = useState(false);

  if (error && !study) return <p className="notice bad" role="alert">{error}</p>;
  if (!study) return <p className="dim">Loading study…</p>;

  async function act(verb: Verb) {
    setBusy(verb);
    setActionError(null);
    try {
      await api.act(id, verb);
      await refresh();
    } catch (e) {
      setActionError(e instanceof ApiError ? e : new ApiError(String(e), 0));
    } finally {
      setBusy(null);
    }
  }
  async function exportReport(kind: "md" | "json" | "copy") {
    try {
      if (kind === "json") return download(`${id}.json`, JSON.stringify(await api.exportJson(id), null, 2), "application/json");
      const md = await api.report(id);
      if (kind === "md") return download(`${id}.md`, md, "text/markdown");
      await navigator.clipboard.writeText(md);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      setActionError(new ApiError(e instanceof Error ? e.message : String(e), 0));
    }
  }

  const b = study.budget, running = study.state === "RUNNING";
  const headline = study.promoted ?? study.best_valid;
  const repro = (experiments.find((e) => e.repro.evaluator_sha256)?.repro ?? {}) as Record<string, unknown>;
  const candidates = experiments.filter((e) => e.kind === "candidate" && e.validation.compared);
  const invalid = candidates.filter((e) => e.state === "INVALID").length;
  const selectedRun = experiments.find((e) => e.id === selected);

  return (
    <div className={`study ${selected ? "with-drawer" : ""}`}>
      <section className="study-head panel">
        <p className="eyebrow">{study.id} · policy <code>{study.policy_arm}</code> · {live === "sse" ? "live" : live === "polling" ? "polling" : "connecting"}</p>
        <h1>{study.goal}</h1>
        <p className="mode-label"><span className="tag strong">{study.mode.toUpperCase()}</span> {study.evidence_label}</p>
        <div className="controls">
          <span className={`pill state-${study.state.toLowerCase()}`}>{study.state}</span>
          {b.approved_spend_usd > 0 && !b.budget_approved && (
            <button className="btn" onClick={() => act("approve-budget")} disabled={!!busy}>Approve ${b.approved_spend_usd.toFixed(2)} cap</button>
          )}
          {study.state === "DRAFT" && <button className="btn primary" onClick={() => act("start")} disabled={!!busy}>▶ Start</button>}
          {(study.state === "PAUSED" || study.state === "FAILED") && <button className="btn primary" onClick={() => act("resume")} disabled={!!busy}>▶ Resume</button>}
          {running && <button className="btn" onClick={() => act("pause")} disabled={!!busy}>Pause</button>}
          {["DRAFT", "RUNNING", "PAUSED", "FAILED"].includes(study.state) && <button className="btn danger" onClick={() => act("cancel")} disabled={!!busy}>Stop study</button>}
          <span className="spacer" />
          <button className="btn" onClick={() => exportReport("md")}>Export report (.md)</button>
          <button className="btn" onClick={() => exportReport("json")}>Export evidence (.json)</button>
          <button className="btn ghost" onClick={() => exportReport("copy")}>{copied ? "Copied" : "Copy report"}</button>
        </div>
        {running && <p className="dim">Pause lets the run in flight finish; Stop cancels it (local runs are killed, external cancellation would be best effort).</p>}
        {study.failure_reason && <p className="notice bad" role="alert"><strong>Study failed:</strong> {study.failure_reason}</p>}
        {actionError && (
          <div className="notice bad" role="alert">{actionError.message}
            {actionError.blockers.length > 0 && <ul>{actionError.blockers.map((x) => <li key={x}>{x}</li>)}</ul>}</div>
        )}
      </section>

      <section className="kpis" aria-label="Summary">
        <div className="kpi hero">
          <p className="eyebrow">{study.promoted ? "Promoted speedup" : "Best valid dev speedup"}</p>
          <p className="kpi-value">{headline ? speedup(headline.metrics.speedup_vs_target) : "—"}</p>
          <p className="dim">{headline
            ? `${configLabel(headline.config)} · ${headline.split} · CI ${speedup(headline.metrics.speedup_ci_low)}–${speedup(headline.metrics.speedup_ci_high)}`
            : candidates.length ? "No candidate has passed the validity gates." : "Not yet run."}</p>
        </div>
        <div className="kpi">
          <p className="eyebrow">Validity</p>
          <p className="kpi-value">{candidates.length ? `${candidates.length - invalid} / ${candidates.length}` : "—"}</p>
          <p className="dim">{candidates.length ? "candidate runs reproduced target-only tokens exactly" : "No candidate scored yet."}</p>
        </div>
        <div className="kpi">
          <p className="eyebrow">Spend</p>
          <p className="kpi-value">{usd(b.observed_spend_usd)}</p>
          <p className="dim">of {usd(b.approved_spend_usd)} approved{b.unknown_cost_events ? ` · ${b.unknown_cost_events} unknown` : ""}</p>
        </div>
        <div className="kpi">
          <p className="eyebrow">Scored runs</p>
          <p className="kpi-value">{b.trials_used}<span className="kpi-of"> / {b.max_trials}</span></p>
          <p className="dim">{duration(b.wall_seconds_used)} measured</p>
        </div>
      </section>

      <div className="columns">
        <section className="panel" aria-labelledby="tl-h">
          <h2 id="tl-h">Experiment timeline</h2>
          <Timeline experiments={experiments} selected={selected} onSelect={setSelected} />
        </section>
        <div className="stack">
          <section className="panel accent" aria-labelledby="why-h">
            <h2 id="why-h">Why Möbius did this</h2>
            <DecisionPanel decisions={decisions} state={study.state} onSelect={setSelected} />
          </section>
          <section className="panel" aria-labelledby="bud-h">
            <h2 id="bud-h">Budget</h2>
            <BudgetGauge b={b} />
          </section>
        </div>
      </div>

      <section className="panel" aria-labelledby="chart-h">
        <h2 id="chart-h">Measured speedup by candidate run</h2>
        <SpeedupChart experiments={experiments} onSelect={setSelected} />
      </section>

      <section className="panel provenance" aria-labelledby="prov-h">
        <h2 id="prov-h">Evidence and provenance</h2>
        <dl>
          <div><dt>Evaluator (frozen)</dt><dd className="mono">{study.frozen_eval_hash?.slice(0, 16) ?? "n/a"}…</dd></div>
          <div><dt>Target model</dt><dd className="mono">{repro.target_model ? `${String(repro.target_model)} · ${String(repro.target_model_sha256).slice(0, 12)}…` : "recorded after the first run"}</dd></div>
          <div><dt>Engine</dt><dd className="mono">{String(repro.engine_version ?? "recorded after the first run")}</dd></div>
          <div><dt>Code</dt><dd className="mono">{repro.git_sha ? `${String(repro.git_sha).slice(0, 10)}${repro.git_dirty ? " (dirty)" : ""}` : "recorded after the first run"}</dd></div>
          <div><dt>Sealed split</dt><dd>Never served by this API. Unevaluated.</dd></div>
        </dl>
      </section>

      {selected && (
        <EvidenceDrawer experimentId={selected} version={`${selectedRun?.state}:${selectedRun?.verdict}:${decisions.length}`}
                        onClose={() => setSelected(null)} onSelect={setSelected} />
      )}
    </div>
  );
}
