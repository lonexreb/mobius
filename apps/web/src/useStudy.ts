import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Decision, type Experiment, type Study } from "./api";

const EVENT_TYPES = [
  "study.created", "study.started", "research.hypothesis", "experiment.queued", "experiment.started",
  "experiment.metrics", "experiment.validated", "research.decision", "budget.updated", "study.paused",
  "study.completed", "source.error",
];

export interface StudyData {
  study: Study | null;
  experiments: Experiment[];
  decisions: Decision[];
  error: string | null;
  live: "sse" | "polling" | "idle";
  refresh: () => Promise<void>;
}

// ponytail: plain fetch + refetch-on-event instead of TanStack Query. One screen, three
// resources; add a query cache when views start sharing data across routes.
export function useStudy(id: string): StudyData {
  const [study, setStudy] = useState<Study | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<StudyData["live"]>("idle");
  const lastEvent = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const [s, e, d] = await Promise.all([api.study(id), api.experiments(id), api.decisions(id)]);
      setStudy(s);
      setExperiments(e);
      setDecisions(d);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [id]);

  useEffect(() => {
    lastEvent.current = 0;
    void refresh();
    let source: EventSource | null = null;
    let poll: number | undefined;
    let pending: number | undefined;
    const onEvent = (ev: MessageEvent) => {
      lastEvent.current = Number(ev.lastEventId) || lastEvent.current;
      window.clearTimeout(pending);
      pending = window.setTimeout(() => void refresh(), 120); // coalesce bursts
    };
    const startPolling = () => {
      setLive("polling");
      poll ??= window.setInterval(() => void refresh(), 3000);
    };
    if ("EventSource" in window) {
      source = new EventSource(api.eventsUrl(id, lastEvent.current));
      source.onopen = () => {
        setLive("sse");
        window.clearInterval(poll);
        poll = undefined;
      };
      source.onerror = startPolling; // EventSource retries itself with Last-Event-ID; poll meanwhile
      EVENT_TYPES.forEach((t) => source!.addEventListener(t, onEvent as EventListener));
    } else {
      startPolling();
    }
    return () => {
      source?.close();
      window.clearInterval(poll);
      window.clearTimeout(pending);
    };
  }, [id, refresh]);

  return { study, experiments, decisions, error, live, refresh };
}
