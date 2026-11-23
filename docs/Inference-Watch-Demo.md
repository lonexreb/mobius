# Demo Möbius Inference Watch in 90 seconds

**The customer is an engineer deciding whether to enable draft decoding.** On the tested M4 Max, the Qwen2.5 7B model running by itself is the code-workload recommendation. One four-token draft setting reproduced only 1 of 8 reference outputs in each of two recordings. Its slower paired throughput ratio was 0.733×. Each brief links the source run IDs and local Autolab evidence. The recommendation applies to the recorded configuration and development prompts.

llama.cpp already documents the technique and ships a speed benchmark. Our deliverable is the configuration decision and a service that changes how it explains the evidence when readers remain undecided. Sundai asks for ongoing work, and Egbe's workshop asks teams to show how contrasting observations change the next action. [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) · [SPEED-Bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/speed-bench/README.md) · [Sundai event](https://luma.com/ktv3iwd9) · [Egbe workshop](https://docs.egbe.ai/events/autonomous-agent-businesses/)

## Start the reader

Use a clean clone of the feature branch if your revival checkout has work in progress. In the branch root run the following. The script uses the project Python environment or installs the existing API dependencies through uv with Python 3.12. WATCH_PYTHON can point to an already provisioned environment.

```sh
./scripts/demo_watch.sh
```

Open [the reader](http://127.0.0.1:8766/) and [the operator view](http://127.0.0.1:8766/loop). Leave the local server running. Requests persist in apps/api/var/reader-demo.sqlite. Closing a tab lets the worker finish. Restarting the server resumes pending work. Keep the receipt URL to return. To rehearse separately, use a fresh file path and a different port, for example:

```sh
WATCH_PORT=8767 WATCH_DB="$PWD/apps/api/var/rehearsal-01.sqlite" ./scripts/demo_watch.sh
```

## Show the change

1. Choose **Code** and **Get my benchmark brief**. Show the target-only recommendation, the eight-prompt denominator and the local Autolab reports. Explain that the service serves recorded measurements.
2. Select **Still undecided**. Use **Request another brief** and repeat once for Code. Show the operator view. Two distinct receipts now trigger a comparison for Code. Reloading a receipt counts it once.
3. Request a third Code brief. It shows both source recordings side by side. Ask a judge what they would run. Click their actual answer. Show the changed presentation and the operator counts. Click **Run Egbe review** to compute the funnel from those local observations.
4. For a full rehearsal, request three more Code briefs. Receipts three through five get the comparison. Receipt six returns to the summary. A Math or Chat request has its own independent state. Record any scripted responses as rehearsal.

## How to read the experiment

SQLite stores each request, frozen presentation, first open and first response. A single background worker builds briefs from the versioned pilot. A transaction assigns up to three comparison receipts per triggered workload. The first two undecided answers are a trigger. Counting them as a control group biases the comparison. The primary business metric is explicit decisions (target only or draft) divided by opened receipts, shown with counts. Show answer rate alongside it. Receipt IDs measure requests and can belong to the same person. The six-receipt rehearsal proves the mechanism.

For the first week, ask five different engineers with the tested setup to try it, record who reached a configuration decision, and count explicit requests for another measurement. Proposed signals are three decisions and two requests for a refresh. With more traffic, preassign eligible new readers 1:1 to summary or comparison, freeze the evidence and define a 24-hour decision window. Keep the trigger cohort outside the estimate. Report both arms' numerators, denominators, nonresponse and uncertainty before adopting either. The local time-limited demo makes a single bounded change; the broader queue experiment is documented in the handoff.

## Sponsor roles, cost and Claude Code

- **Autolab** supplies the evaluator format and local signed reports for the frozen reference and rejected draft. Open either evidence link in the reader. New hardware or models require fresh measurements. Hosted scoring is a separate account action. [Autolab hills](https://docs.autolab.ai/guides/hills/)
- **Egbe** supplies the official offline funnel analyzer. The operator button feeds it observed local receipt events. The launcher looks for the installed Codex plugin helper. On another machine, set WATCH_EGBE_ANALYZER to the official plugin's skills/egbe-os-analytics/scripts/umami_funnel.py. The button reports a missing helper explicitly. Local analysis does not send reader events to a hosted account. [Egbe plugin](https://github.com/egbe-ai/egbe-plugins)
- **Claude Code** can review the demo with the project's /watch-review command. Start Claude Code in this feature branch while the local server is running. The optional Egbe connector uses its own sign-in. Egbe documents the following terminal flow, then /egbe:egbe-setup in a fresh Claude Code session. Its setup can enable project session logging. Confirm the business, project and Claude telemetry scope with the owner before completing that flow. The website's Connectors panel is a separate connection surface. [Egbe installation and setup](https://github.com/egbe-ai/egbe-plugins#install)

```sh
claude plugin marketplace add egbe-ai/egbe-plugins
claude plugin install egbe@egbe
```

The local demo requires no hosted API key, new GPU run or per-request model spend. Claude Code usage follows the user's existing plan. The live business pilot needs a reachable host and verified analytics for actual readers. Ask before paid hosting, publishing, or creating any recurring worker or review. A foreground process on this Mac serves new requests while it is running and the Mac is awake. No ongoing hosted service has been activated by this walkthrough.

## Verify from the branch

```sh
python3 apps/inference_watch/test_worker.py
apps/api/.venv/bin/python apps/inference_watch/test_demo.py
```

Use the Python selected by WATCH_PYTHON for the second command if the environment lives elsewhere. Check /api/loop for distinct receipt counts and /api/egbe-report for local requesting and activated counts. A fresh demonstration database should return zero counts before the first reader.
