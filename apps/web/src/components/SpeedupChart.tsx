import { useState } from "react";
import type { Experiment } from "../api";
import { configLabel, runStatus, speedup, tokenMatch } from "../format";

interface Props {
  experiments: Experiment[];
  onSelect: (id: string) => void;
}

const W = 720, H = 280, PAD = { top: 20, right: 24, bottom: 46, left: 52 };

/** Paired speedup per scored candidate run, with its bootstrap CI, against the 1.00× target-only line.
 *  Runs without a measured speedup are not plotted; they are never drawn at zero. */
export function SpeedupChart({ experiments, onSelect }: Props) {
  const [showInvalid, setShowInvalid] = useState(true);
  const [hover, setHover] = useState<string | null>(null);
  const scored = experiments.filter((e) => e.kind === "candidate" && e.metrics.speedup_vs_target !== undefined);
  const shown = scored.filter((e) => showInvalid || e.state === "VALID");

  if (scored.length === 0) {
    return (
      <div className="empty">
        <p>No scored candidates yet.</p>
        <p className="dim">Each candidate appears here with its measured speedup and confidence interval once its run finishes.</p>
      </div>
    );
  }

  const values = shown.flatMap((e) => [e.metrics.speedup_ci_low!, e.metrics.speedup_ci_high!, e.metrics.speedup_vs_target!]);
  const lo = Math.floor((Math.min(0.9, ...values) - 0.02) * 10) / 10;
  const hi = Math.ceil((Math.max(1.1, ...values) + 0.02) * 10) / 10;
  const tickStep = hi - lo > 1.2 ? 0.2 : 0.1;
  const y = (v: number) => PAD.top + ((hi - v) / (hi - lo)) * (H - PAD.top - PAD.bottom);
  const step = (W - PAD.left - PAD.right) / Math.max(shown.length, 1);
  const x = (i: number) => PAD.left + step * (i + 0.5);
  const ticks = Array.from({ length: Math.round((hi - lo) / tickStep) + 1 }, (_, i) => lo + i * tickStep);
  const active = shown.find((e) => e.id === hover);

  return (
    <div className="chart">
      <div className="chart-head">
        <ul className="legend" aria-label="Legend">
          <li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" className="mark-valid" /></svg>Passed validity gates</li>
          <li><svg width="14" height="14" aria-hidden="true"><path d="M3 3l8 8M11 3l-8 8" className="mark-invalid" /></svg>Failed a validity gate</li>
          <li><svg width="14" height="14" aria-hidden="true"><path d="M7 1v12M3 1h8M3 13h8" className="whisker" /></svg>95% CI (within-run)</li>
        </ul>
        <label className="toggle">
          <input type="checkbox" checked={showInvalid} onChange={(e) => setShowInvalid(e.target.checked)} />
          Show invalid runs
        </label>
      </div>
      <div className="chart-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label={`Speedup versus target-only for ${shown.length} candidate runs. Details are in the run list.`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
              <text x={PAD.left - 8} y={y(t) + 4} textAnchor="end" className="tick">{t.toFixed(2)}×</text>
            </g>
          ))}
          <line x1={PAD.left} x2={W - PAD.right} y1={y(1)} y2={y(1)} className="refline" />
          <text x={W - PAD.right} y={y(1) - 6} textAnchor="end" className="reflabel">target-only = 1.00×</text>
          {shown.map((e, i) => {
            const m = e.metrics, valid = e.state === "VALID", cx = x(i), cy = y(m.speedup_vs_target!);
            return (
              <g key={e.id} className={`pt ${hover === e.id ? "is-hover" : ""}`} tabIndex={0} role="button"
                 aria-label={`${e.id}, ${configLabel(e.config)}, ${e.split}, speedup ${speedup(m.speedup_vs_target)}, ${runStatus(e).text}`}
                 onMouseEnter={() => setHover(e.id)} onMouseLeave={() => setHover(null)}
                 onFocus={() => setHover(e.id)} onBlur={() => setHover(null)}
                 onClick={() => onSelect(e.id)} onKeyDown={(k) => k.key === "Enter" && onSelect(e.id)}>
                <rect x={cx - step / 2} y={PAD.top} width={step} height={H - PAD.top - PAD.bottom} className="hit" />
                <path d={`M${cx} ${y(m.speedup_ci_low!)}V${y(m.speedup_ci_high!)}M${cx - 5} ${y(m.speedup_ci_low!)}h10M${cx - 5} ${y(m.speedup_ci_high!)}h10`} className="whisker" />
                {valid
                  ? <circle cx={cx} cy={cy} r={e.verdict === "promoted" ? 7 : 5} className="mark-valid" />
                  : <path d={`M${cx - 5} ${cy - 5}l10 10M${cx + 5} ${cy - 5}l-10 10`} className="mark-invalid" />}
                <text x={cx} y={H - PAD.bottom + 16} textAnchor="middle" className="tick">#{e.seq}{e.split === "audit" ? " audit" : ""}</text>
                <text x={cx} y={H - PAD.bottom + 30} textAnchor="middle" className="tick dim">
                  L{e.config?.lookahead_tokens} p{e.config?.draft_p_min}
                </text>
              </g>
            );
          })}
        </svg>
        {active && (
          <div className="tooltip" role="status"
               style={{ left: `${(x(shown.indexOf(active)) / W) * 100}%`, top: `${(y(active.metrics.speedup_ci_high!) / H) * 100}%` }}>
            <strong>{active.id}</strong> · {configLabel(active.config)}
            <dl>
              <dt>Speedup</dt><dd>{speedup(active.metrics.speedup_vs_target)}</dd>
              <dt>95% CI</dt><dd>{speedup(active.metrics.speedup_ci_low)} – {speedup(active.metrics.speedup_ci_high)}</dd>
              <dt>Tokens matched</dt><dd>{tokenMatch(active)}</dd>
              <dt>Status</dt><dd>{runStatus(active).text}</dd>
            </dl>
          </div>
        )}
      </div>
      <p className="caption">
        Speedup = geometric mean over code, math and chat of median decode tokens/s, candidate ÷ interleaved target-only control.
        Pilot scale (8 prompts per domain). Click a point for its evidence.
      </p>
    </div>
  );
}
