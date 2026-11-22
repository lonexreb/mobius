import { useCallback, useEffect, useState } from "react";
import { api, type Connections, type Study } from "./api";
import { GoalComposer } from "./components/GoalComposer";
import { StudyPage } from "./components/StudyPage";
import { speedup, usd } from "./format";

const CONNECTION_LABELS: [keyof Connections, string][] = [
  ["local", "Local evaluator"], ["mobius_cli", "Möbius Rust optimizer"], ["autolab", "Autolab"],
  ["gpu", "CUDA GPU"], ["llm_proposer", "LLM proposer"], ["egbe", "Egbe"],
];

// The study id lives in the URL hash so a study view can be linked and reloaded.
function useHashRoute(): [string | null, (id: string | null) => void] {
  const read = () => window.location.hash.match(/^#\/study\/([\w-]+)$/)?.[1] ?? null;
  const [id, setId] = useState<string | null>(read);
  useEffect(() => {
    const onHash = () => setId(read());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return [id, (next) => { window.location.hash = next ? `#/study/${next}` : "#/"; }];
}

export function App() {
  const [studyId, go] = useHashRoute();
  const [studies, setStudies] = useState<Study[]>([]);
  const [connections, setConnections] = useState<Connections | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    Promise.all([api.studies(), api.connections()])
      .then(([s, c]) => { setStudies(s); setConnections(c); setError(null); })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load, studyId]);

  const current = studies.find((s) => s.id === studyId);
  return (
    <>
      <header className="topnav">
        <a className="brand" href="#/" aria-label="Möbius Lab home"><span aria-hidden="true">∞</span> MÖBIUS LAB</a>
        <nav aria-label="Main navigation"><a href="#/" aria-current={studyId ? undefined : "page"}>Research projects</a></nav>
        <p className="nav-status">
          <span className={`led ${connections?.local.ready ? "on" : "off"}`} aria-hidden="true" />
          Compute: {connections ? (connections.local.ready ? "local only" : "unavailable") : "checking…"}
          <span className="sep">·</span>GPU: not connected
          {current && <><span className="sep">·</span><span className="mono">{usd(current.budget.observed_spend_usd)} / {usd(current.budget.approved_spend_usd)}</span></>}
        </p>
      </header>

      <main>
        {error && <p className="notice bad" role="alert">{error} <button className="link" onClick={load}>Retry</button></p>}
        {studyId ? <StudyPage key={studyId} id={studyId} /> : (
          <div className="home">
            <section className="lede">
              <p className="eyebrow">Evidence-calibrated research director</p>
              <h1>Anyone can automate an experiment. <em>Möbius decides which evidence to trust.</em></h1>
              <p>Give it an objective and a budget. It proposes experiments, scores them with a fixed evaluator, and then
                promotes, replicates, audits, rejects or stops, with the reason on the record.</p>
            </section>
            <GoalComposer studies={studies} connections={connections} onCreated={(s) => go(s.id)} />
            <aside className="side">
              <section className="panel" aria-labelledby="conn-h">
                <h2 id="conn-h">Connections</h2>
                <ul className="conns">
                  {connections ? CONNECTION_LABELS.map(([key, label]) => (
                    <li key={key}>
                      <span className={`led ${connections[key].ready ? "on" : "off"}`} aria-hidden="true" />
                      <span><strong>{label}</strong> {connections[key].ready ? "ready" : "not connected"}
                        {connections[key].blocker && <small>{connections[key].blocker}</small>}</span>
                    </li>
                  )) : <li className="dim">Checking…</li>}
                </ul>
              </section>
              <section className="panel" aria-labelledby="recent-h">
                <h2 id="recent-h">Studies</h2>
                {studies.length === 0 ? <p className="dim">No studies yet. Create one to get a first baseline.</p> : (
                  <ul className="studies">
                    {studies.map((s) => (
                      <li key={s.id}>
                        <a href={`#/study/${s.id}`}>
                          <span className="mono">{s.id}</span>
                          <span className={`pill state-${s.state.toLowerCase()}`}>{s.state}</span>
                          <span className="dim">{s.policy_arm} · {s.mode.toUpperCase()} · {s.budget.trials_used} runs</span>
                          <span>{s.promoted ? `Promoted ${speedup(s.promoted.metrics.speedup_vs_target)}`
                            : s.latest_decision ? `Last: ${s.latest_decision.action.replace(/_/g, " ")}` : "Not started"}</span>
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </aside>
          </div>
        )}
      </main>
      <footer>
        <p>Pilot instrumentation. Local results are not comparable to the Qwen3-8B / EAGLE-3 CUDA flagship, which has not been run. No statistical significance is claimed.</p>
      </footer>
    </>
  );
}
