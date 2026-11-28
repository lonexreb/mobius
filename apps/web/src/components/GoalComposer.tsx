import { type FormEvent, useEffect, useState } from "react";
import { api, ApiError, type Connections, type Mode, type Study, type Template } from "../api";

interface Props {
  studies: Study[];
  connections: Connections | null;
  onCreated: (study: Study) => void;
}

const ARMS: Record<string, string> = {
  evidence_aware: "Replicates borderline gains, audits before promoting, stops on systematic failure.",
  metric_greedy: "Same proposer; promotes the highest dev score. Validity gates only.",
  tpe: "Configs from the Rust Möbius TPE optimizer; promotes the highest dev score.",
  random: "Seeded random configs from the pinned search space.",
  fixed_spec: "Baseline plus one pre-registered fixed config.",
  target_only: "Baseline only.",
};
const MODES: { id: Mode; title: string; note: string }[] = [
  { id: "local", title: "Local validation", note: "Real measurements on this machine. $0." },
  { id: "modal", title: "Modal GPU (paid)", note: "Qwen3-8B + EAGLE-3 on a rented A100. Needs a cap." },
  { id: "autolab", title: "Autolab project", note: "Externally scored runs. Not connected." },
  { id: "replay", title: "Replay recorded runs", note: "Re-serve a finished study. Nothing re-measured." },
];
const modelName = (m: Template["target_model"]) => (typeof m === "string" ? m : `${m.repo}@${m.revision.slice(0, 8)}`);

export function GoalComposer({ studies, connections, onCreated }: Props) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateId, setTemplateId] = useState("local_speculative_qwen25");
  const [goal, setGoal] = useState("Improve decode throughput with speculative decoding without changing greedy outputs.");
  const [mode, setMode] = useState<Mode>("local");
  const [arm, setArm] = useState("evidence_aware");
  const [trials, setTrials] = useState(8);
  const [spend, setSpend] = useState(0);
  const [minutes, setMinutes] = useState(60);
  const [replayOf, setReplayOf] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    api.templates().then((t) => setTemplates(t.templates)).catch((e) => setError(e));
  }, []);

  const template = templates.find((t) => t.id === templateId);
  const replayable = studies.filter((s) => s.mode !== "replay" && s.budget.trials_used > 0);
  const flagship = templateId === "qwen3_speculative_ood";
  const blocker =
    flagship && mode === "local" ? "The Qwen3-8B flagship needs CUDA + vLLM: choose Modal GPU (paid) or Autolab."
    : mode === "modal" && !flagship ? "Modal GPU mode runs the Qwen3-8B flagship template only."
    : mode === "modal" && spend <= 0 ? "Modal is paid compute: set a spend cap above $0."
    : mode === "local" && connections && !connections.local.ready ? `Local evaluator unavailable: ${connections.local.blocker}`
    : mode === "replay" && !replayOf ? "Choose a finished study to replay."
    : arm === "tpe" && connections && !connections.mobius_cli.ready ? `TPE arm unavailable: ${connections.mobius_cli.blocker}`
    : null;
  const willStart = mode !== "autolab" && spend === 0;

  async function submit(ev: FormEvent) {
    ev.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const study = await api.create({
        template_id: templateId, goal, mode, policy_arm: arm,
        budget: { max_trials: trials, approved_spend_usd: spend, max_wall_seconds: minutes * 60 },
        ...(mode === "replay" ? { replay_of: replayOf } : {}),
        notes: "Pilot only: no statistical significance claim",
      });
      onCreated(willStart ? await api.act(study.id, "start") : study);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="composer panel" onSubmit={submit} aria-labelledby="composer-h">
      <h2 id="composer-h">New research study</h2>
      <label className="field">
        <span>Research objective</span>
        <textarea value={goal} onChange={(e) => setGoal(e.target.value)} rows={2} minLength={5} maxLength={500} required />
      </label>

      <div className="grid-2">
        <label className="field">
          <span>Template</span>
          <select value={templateId} onChange={(e) => setTemplateId(e.target.value)}>
            {templates.map((t) => <option key={t.id} value={t.id}>{t.title}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Decision policy</span>
          <select value={arm} onChange={(e) => setArm(e.target.value)}>
            {Object.keys(ARMS).map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          <small>{ARMS[arm]}</small>
        </label>
      </div>

      {template && (
        <dl className="facts">
          <div><dt>Target</dt><dd>{modelName(template.target_model)}</dd></div>
          <div><dt>Draft</dt><dd>{modelName(template.draft_model)}</dd></div>
          <div><dt>Engine</dt><dd>{template.engine}</dd></div>
          <div><dt>Prompts</dt><dd>{template.sources.map((s) => `${s.domain} ${s.repo}@${s.revision.slice(0, 7)}`).join(" · ")}</dd></div>
          <div><dt>Metric</dt><dd>{template.primary_metric}</dd></div>
          <div><dt>Validity</dt><dd>{template.validity}</dd></div>
          <div className="wide"><dt>Status</dt><dd>{template.label}</dd></div>
        </dl>
      )}

      <fieldset className="modes">
        <legend>Mode</legend>
        {MODES.map((m) => (
          <label key={m.id} className={`mode ${mode === m.id ? "is-on" : ""}`}>
            <input type="radio" name="mode" value={m.id} checked={mode === m.id} onChange={() => setMode(m.id)} />
            <strong>{m.title}</strong>
            <small>{m.note}</small>
          </label>
        ))}
      </fieldset>

      {mode === "replay" && (
        <label className="field">
          <span>Recorded study</span>
          <select value={replayOf} onChange={(e) => setReplayOf(e.target.value)}>
            <option value="">{replayable.length ? "Select a study…" : "No finished studies to replay yet"}</option>
            {replayable.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.policy_arm} · {s.budget.trials_used} runs</option>)}
          </select>
        </label>
      )}

      <div className="grid-3">
        <label className="field">
          <span>Max scored runs</span>
          <input type="number" min={1} max={24} value={trials} onChange={(e) => setTrials(Number(e.target.value))} />
          <small>Baselines, replicates and audits all count.</small>
        </label>
        <label className="field">
          <span>Spend cap (USD)</span>
          <input type="number" min={0} max={1000} step={1} value={spend} onChange={(e) => setSpend(Number(e.target.value))} />
          <small>{spend > 0 ? "Needs explicit approval before any run." : "Default $0: nothing billed."}</small>
        </label>
        <label className="field">
          <span>Wall-time cap (min)</span>
          <input type="number" min={1} max={1440} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} />
        </label>
      </div>

      <p className="intent">
        <strong>Intent:</strong> run up to {trials} scored evaluations with the <code>{arm}</code> policy in{" "}
        <code>{mode}</code> mode, spending at most ${spend.toFixed(2)}.{" "}
        {mode === "autolab" && "Creates a draft only: remote dispatch stays blocked until Autolab and a GPU are connected and the budget is approved."}
        {mode === "local" && "Runs llama.cpp on this machine for a few minutes."}
        {mode === "modal" && `Creates a draft. Each run reserves up to $${template?.estimated_max_usd_per_run?.toFixed(2) ?? "?"} against the cap; you approve the cap, then start.`}
        {mode === "replay" && "Re-serves recorded artifacts; the controller decides again."}
      </p>

      {blocker && <p className="notice warn" role="status">{blocker}</p>}
      {error && (
        <div className="notice bad" role="alert">
          {error.message}
          {error.blockers.length > 0 && <ul>{error.blockers.map((b) => <li key={b}>{b}</li>)}</ul>}
        </div>
      )}
      <button className="btn primary" type="submit" disabled={busy || !!blocker}>
        {busy ? "Working…" : willStart ? "Create and start" : "Create draft"}
      </button>
    </form>
  );
}
