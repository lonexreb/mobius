# Möbius Inference Watch
## Claude Code handoff · Sundai San Francisco · November 22, 2026

**Build a service for engineers running Qwen on Apple Silicon.** A reader chooses code, math, or chat and gets a short answer to “Should I enable speculative decoding on this setup?” Each answer includes the target, draft, engine build, hardware, speed ratio, exact-output counts, source runs, and date. Reader demand chooses which workload receives the next evaluation slot. A failed experiment is a useful deliverable when it prevents a bad deployment.

**Demo boundary:** one hardware profile, three workload choices, a read-only customer brief, one background worker, and one feedback decision. The proposed paid offer is a **$9/month watch for one profile** with refreshed evidence after relevant changes. This price is a hypothesis. Test repeat use and willingness to pay before adding billing.

### Start Claude Code

In the existing RSI clone, use the following commands. If this checkout has ongoing work, use a separate clone of the new branch. The pilot ships with it.

```sh
git fetch origin hack/sundai-inference-watch
git switch hack/sundai-inference-watch
claude
```

Paste this into Claude Code as one message:

```text
/goal Deliver the Sundai Inference Watch workflow on hack/sundai-inference-watch. Read docs/SUNDAI_INFERENCE_WATCH.md first. Start with the existing worker, measured pilot, local Autolab hill and Egbe rehearsal. Implement one reader request through brief delivery, record Egbe activation after confirmed delivery, use observed demand to select the next bounded evaluation, and verify that a later tick runs after closing the browser and restarting the worker. Follow the completion tests and cost gates below. Use the existing UI only where it saves time. Produce a runbook, tests, two contrasting decision traces, the served brief, the next scheduled time and the exact status of each sponsor. Missing authentication must leave a precise Codex request and a running local path. Make small commits on this branch. Keep external spending at $0 unless the owner approves a specified amount and activity. Begin by running the rehearsal, then implement.
```

Claude Code 2.1.330 was present on the development Mac. The goal belongs inside the interactive session. [Claude Code command reference](https://code.claude.com/docs/en/commands)

## Why this fits the challenge

The organizers ask for projects that autonomously operate and improve a process. The event lists a 6 p.m. Pacific check-in and presentations. Egbe’s workshop asks for two contrasting observations, the action each produces, and a way to tell whether it helped. A saved loop needs a verified runner. Egbe credits are promised to participants; the event specifies no dollar value. [Organizer listing](https://luma.com/ktv3iwd9) · [Egbe’s workshop for this event](https://docs.egbe.ai/events/autonomous-agent-businesses/)

I directly inspected three same-day Sundai slides from the desktop. One distinguishes building, operating and continuously improving a company. Another asks for background work on the project after the demo and illustrates processing new data and deploying updates. The third calls for closed business loops with human oversight. The referenced chat omits all five attachment binaries. The original chat also described skill nodes, spherical routing, status and KPIs. Those design cues are secondary evidence. The functional interpretation is one visible decision with its evidence, resulting job, and outcome.

## What the branch already contains

| Source inspected | Finding and reuse decision |
|---|---|
| [GitHub revival branch](https://github.com/lonexreb/mobius/tree/revival/sundai-recursive-experiments-2026) | Remote tip at inspection was `6eedc12`. The RSI checkout contained five additional commits through `a4f73a3`, including a functioning API, UI, ledger, evaluator, and runbook. This hack branch descends from the local tip. Remote main at inspection was `0417470`. |
| [Zigsaw core graph](https://github.com/lonexreb/zigsaw/blob/22e54b2bb8b91966eacf385b3096319200a4063f/frontend/src/components/KnowledgeGraph3D.tsx) · [router](https://github.com/lonexreb/zigsaw/blob/22e54b2bb8b91966eacf385b3096319200a4063f/frontend/src/components/nodes/RouterNode.tsx) | The sphere exists. The graph uses Three.js sphere geometry and a 3D force graph. The configurable React Flow router is a separate component. Borrow the visual language of the active path. A three-node strip and an evidence drawer fit the day. |
| [Zigsaw Labs canvas](https://github.com/lonexreb/zigsaw-labs/tree/b53eb2d16519b9e5bac31d330a94426c856d2f21/apps/frontend/src/components/canvas) | Includes decision, metrics, publishing and quality-control nodes. Rebuilding its campaign and worker services would put the effort in the editor. |
| [Zigsaw YC editor](https://github.com/lonexreb/zigsaw-yc/blob/40047d62c7aa7bebeed59c27766fbde79307bca5/frontend/src/WorkflowEditor.tsx) | React Flow editor, node library and minimap. The customer needs a brief and one choice. Use the existing Möbius web app. |

The signed local Autolab reports, the Egbe rehearsal counts and the code brief are under `docs/evidence/inference-watch/`. Both Autolab report signatures verified locally. Run `python3 apps/inference_watch/test_worker.py` to exercise the fresh-process recovery and duplicate-request tests.

Read `docs/MOBIUS_LAB_RUNBOOK.md` for the existing app. The app binds to loopback, has one API owner per ledger, and currently pauses interrupted studies. Its Autolab endpoint imports records; it cannot launch a climb. Local recorded imports do not prove sponsor authentication. The older frontier handoff is background for the research engine. Today's deliverable has no dependency on a Qwen3/EAGLE-3 GPU run.

| Ready in this branch | Evidence |
|---|---|
| `experiments/inference-watch/` | A working local Autolab hill, two example submissions, and a frozen table of measured outcomes. The sponsor CLI checked the manifest and ran three passing hill tests. Its local evaluator accepted the code reference at 1.0× and rejected `d4-p75` for token mismatch. Frozen tree at verification was `1c19fb7655711dbe1b8c590ab474b74b65563152`. No hosted climb ran. |
| `scripts/export_inference_watch_pilot.py` | Reproduces the shipped aggregate JSON byte-for-byte from the two original study directories. Checks control tokens against target-only output and recomputes per-domain ratios. Run IDs and artifact hashes travel with the pilot; the raw model outputs and machine paths stay local. |
| `apps/inference_watch/worker.py` | Standard-library SQLite request queue. One worker tick selects the workload with the most outstanding requests and fulfills that group atomically. FIFO breaks ties. Results persist across process restarts and resubmitting an ID with another workload fails. The worker calls the hill’s scoring function directly for fast recorded briefs. A separately run sponsor CLI produces the signed evaluation report. |
| `scripts/check_egbe_watch.sh` and `apps/inference_watch/rehearsal/egbe-events.json` | Live public Egbe contract verified. The official offline analysis helper counted two included synthetic sessions, two clicks and one activation, and excluded one smoke session. Every row is a fixture. No business key, site registration, customer event, or hosted analytics read has been verified. |
| Existing backend | Reran 85 API/evaluator tests on the local base. All passed with one dependency deprecation warning. |

## Run the working slice now

These commands run from the branch root with Python 3.11+ and need no model downloads, network account, or GPU. The example uses the ignored local state directory. Use a separate database for customer traffic.

```sh
W=apps/inference_watch/worker.py
DB=apps/api/var/watch-rehearsal.sqlite
python3 "$W" --db "$DB" request demo-m1 math
python3 "$W" --db "$DB" request demo-c1 code
python3 "$W" --db "$DB" request demo-c2 code
python3 "$W" --db "$DB" watch --interval 2
```

In another terminal, read a completed result:

```sh
python3 apps/inference_watch/worker.py --db apps/api/var/watch-rehearsal.sqlite brief demo-c1
```

The first two ticks should serve code and then math. Stop the worker with Ctrl-C, resubmit an identical request, and run `tick` in place of `watch --interval 2`. Expect `idle` and the same stored brief. Use fresh IDs for a second run. The worker can run with the browser closed; the foreground command still needs a live process and an awake host. The verification run was stopped after its checks. A persistent job is an acceptance item below.

To repeat the sponsor scoring, follow `experiments/inference-watch/README.md`. Run the CLI in a scratch directory. Keep both JSON reports and verify the `passed` field, metric and tree hash. Both an accepted and a rejected submission can produce a report. For local testing only, `autolab hills new/check/commit/eval` worked with a signed-out CLI. A sponsor-hosted run needs additional setup below. [Hill format and commands](https://docs.autolab.ai/guides/hills/)

To reproduce the Egbe rehearsal, use the installed official analysis helper, or obtain its public source in a scratch directory. The reviewed helper SHA-256 was `2bdb26ce7829e9e64e869925587e49853ed2a231c0521e354e4e15171d8fb686`.

```sh
./scripts/check_egbe_watch.sh
mkdir -p apps/api/var/egbe-rehearsal
curl -fsS https://os.hz.egbe.dev/skills/egbe-os-analytics/resources/scripts/umami_funnel.py -o apps/api/var/egbe-rehearsal/umami_funnel.py
# Check the downloaded helper against the reviewed digest before running it.
python3 apps/api/var/egbe-rehearsal/umami_funnel.py --from-events apps/inference_watch/rehearsal/egbe-events.json
```

## Product and engineering work for Claude

**One reader journey.** Show one supported environment, the measurement date, and three workload buttons. “Get my benchmark brief” creates an idempotent request and returns an opaque receipt. The reader can revisit it after closing the tab. Show the target-only recommendation when all drafts fail, an evidence table, and one further action, “Check this workload again when evidence changes.” A saved watch signals interest. Revenue requires a settled payment. [Egbe product analytics](https://docs.egbe.ai/infrastructure/analytics-and-email/)

1. **Connect the reader to the queue (30 minutes).** Factor the existing worker operations so FastAPI and the CLI use the same implementation. Add `POST /watch/requests` with an idempotency key and allowlisted workload, `GET /watch/requests/{receipt}` and an opt-in watch operation. Generate an unguessable receipt server-side and store its hash. Separate the local operator dashboard from the public surface. Expose only aggregate brief fields. Keep the existing admin, jobs, import, budget and lab endpoints on loopback. Poll the receipt for completion. Bind a demo on localhost first.
2. **Close the customer-to-experiment loop (30 minutes).** The starter already prioritizes outstanding work and serves recorded evidence. Persist a versioned evidence catalog and `next_check_at`. Each eligible cycle ranks distinct waiting watches or newly requested workloads with stale evidence, selects at most one domain, and records the queue snapshot, policy version and reason. No stale domain or new version means sleep until the next check. Pending work older than two service intervals gets the next slot. The action must **execute** the selected check and update its result. Use the local Autolab evaluator immediately. If only frozen inputs exist, update “last replayed” and leave “last measured” intact. Use an approved new local paired evaluation when engine/model/evaluator or supported dataset revisions change; index the new artifacts and rerun the fixed hill. A replay cannot satisfy a request for fresh measurements.
3. **Wire Egbe into the operating decision (30 minutes).** The product backend confirms receipt delivery. The browser emits `cta_click` on request and `activated` after a ready brief is actually shown, with `activation_type=benchmark_brief_viewed` and an opaque activation ID. Track watch opt-in separately. Preserve first-touch attribution. Retrieve the last complete reporting window and reconcile activations with issued receipts. When at least three distinct qualified readers request fresh evidence for one domain, allocate the next eligible check there. If requests accumulate while readers fail to receive briefs, spend the slot retrying or repairing delivery before refreshing the model evidence. Use a separate rehearsal namespace with two contrasting observations while the project connection is pending. An analytics outage is missing coverage; it must not erase durable requests.
4. **Prove continuity and demo it (25 minutes).** Run one worker per durable queue under a supervised process. On this Mac a user LaunchAgent can invoke one bounded `tick` every 60 s using absolute paths to the Python executable, script and persistent DB; set `StartInterval=60`, `RunAtLoad=true`, explicit log paths and a unique Label. Put the plist template and load/unload instructions in the runbook. Get the owner’s agreement before installing the ongoing job. Verify registration with `launchctl print gui/$(id -u)/<label>` and verify execution from a post-registration timestamp. Mac sleep defers work. For laptop-independent operation, use an authorized always-on runner with durable storage. Verify Egbe hosting persistence and scheduling support before moving the queue there. Only one instance owns the schedule. [Egbe recurring work](https://docs.egbe.ai/guides/continuity/)
5. **Spend any remaining time on evidence readability.** Reuse the current React evidence drawer and render Request → Evaluate → Deliver with timestamps, pending work count, last decision, next tick, and source ID. The 3D graph, general workflow editor, arbitrary models, outbound marketing, payments, multi-agent orchestration, and a GPU climb can wait.

**Persistence contract.** Keep the existing study ledger as the research record. Put reader requests, consented watches, signal cursors, brief versions, jobs and decisions in the separate watch store. Minimum uniqueness constraints are request idempotency key, one `(workload, input_hash, evaluator_version, check_type)` per job, and one activation per receipt/version. Job states are pending, leased, succeeded, retryable and blocked. Claim under a short SQLite transaction; release the database lock during external work. Renew or expire leases, cap transient retries at three with backoff, and reconcile an ambiguous sponsor write by saved run ID before another submission. Commit the result and the next eligible action together. Store a redacted sponsor request ID and explicit evidence class. Save a cost reservation before dispatch and reconcile actual charge once. No periodic idle submissions.

## Experiment design and measures

**Technical question.** For the stated machine and build, can a draft improve decoding speed while returning identical greedy token IDs? Freeze model hashes, engine version, prompt split, generation settings and evaluator revision. The existing local benchmark pairs candidate and target-only runs under the same conditions. Throughput is candidate/control median decode tokens per second within workload. The replay hill takes the smaller ratio across both recordings and disqualifies any draft that diverges on any prompt. Repetition used the same eight prompts per domain. The public pilot is development evidence.

**Measured starting point.** In the local code pilot, `d4-p75` matched **1/8** in each recording and ran at **0.733×** and **0.748×** the interleaved reference. All three repeated settings were slower and failed correctness for code. The existing all-domain ledger has ten candidate executions across two policies, with zero passing exactness. Three settings overlap across the two studies. One controller stopped at four total runs; the other used eight. This single comparison supplies neither a generalized policy lift nor a measured dollar saving.

**Gate for a fresh recommendation.** Declare one candidate and primary workload before unlocking the audit. Require exact equality, a reproduced target control, and a proposed practical threshold of ≥5% speedup. Replicate in a separate session, counterbalance run order and verify on a fresh longer-prompt set with recorded hashes. Require the lower bound of the paired 95% bootstrap interval to exceed 1.0 in each session. Cluster by prompt, do not count repeated tokens as independent observations, and keep the sealed split out of tuning. If a gate fails or the hardware profile changes, serve the existing reference with a dated caveat. No evidence on CUDA or Qwen3 follows from this pilot.

| Business question | Metric and treatment |
|---|---|
| Does allocating the next check by demand help? | Primary: fraction of eligible waiting receipts served a matching fresh brief within one check slot. Numerator is qualifying receipts served by the deadline; denominator is unique eligible receipts at its start. Track mean ticks, p95 wait and oldest wait by domain. Frozen replays count only for a recorded-brief request. |
| Did readers get value? | Ordered activation = distinct eligible first-time reader receipts confirmed ready **and then viewed** within 24 h / distinct eligible first-time request receipts with a complete 24 h window. Reconcile a durable delivery with an Egbe session event. Show numerator and denominator, missing coverage, excluded operator/smoke traffic and dates. Egbe’s general stage counts alone do not establish event order. |
| Will they return? | D7 = consenting activated watchers who open a later brief version within seven days / activated watchers whose full seven-day window elapsed. Track watch opt-ins separately. An empty denominator is “insufficient data.” Intent at the proposed $9 price is distinct from collected revenue. |
| Can it operate? | Track scheduled versus completed eligible ticks, last successful tick, queue age, duplicates, unknown-cost actions blocked, false positives and incremental cost per delivered brief. Use charged sponsor/host costs plus known metered costs; label unmetered electricity and unknowns. |

**Fair comparison.** Freeze the observed arrival tape and compare demand-first with FIFO using identical capacity, requests, costs, initial catalog and deadlines. The verified queue rehearsal had math arrive first and code twice next. Demand-first served 2/3 in the first slot, then 1/3. FIFO would serve 1/3, then 2/3. Mean waiting time was 1.33 versus a calculated 1.67 ticks. These are synthetic arrivals and a counterfactual FIFO schedule. They demonstrate the mechanism. They cannot establish customer demand or a production effect.

**Live policy trial.** Once enough real traffic exists, prospectively assign nonoverlapping comparable service blocks to FIFO or demand-first in balanced random order; assign at the block level because readers share one queue. Freeze a common snapshot or drain carryover at each boundary and log exclusions. Match capacity and time of day. Predeclare at least 20 independent paired blocks and 50 eligible requests per arm as a pilot floor; calculate power from observed block variability before a definitive test. Estimate the paired difference with a block-level bootstrap interval. Promote only if the point gain is ≥10 percentage points, its 95% interval excludes zero, and p95 wait and oldest-workload guardrails both pass (p95 no worse by more than one tick; no eligible workload waits more than two intervals before priority override). Review once at the fixed horizon. Low traffic or an inconclusive interval keeps the prior policy. Repeat use and willingness to pay remain separate product questions.

## Accounts, APIs, credits and cost

| Dependency | What to configure and verify | Current state and cost |
|---|---|---|
| Local pilot | Python 3.11+, SQLite in stdlib, recorded catalog. The old UI additionally uses Node, npm, uv and Rust via `./scripts/demo_local.sh`. Fresh local ML needs the pinned model files, prompt data and llama.cpp listed in the lab runbook. | Request worker and local sponsor scoring verified. $0 incremental external charge incurred. Hardware and electricity are unmetered. |
| Autolab | Install official CLI using its setup guide; run `autolab --version` and `autolab whoami`. For a hosted climb the owner signs in via `autolab login`. Use a dedicated submission folder, publish only agreed hill material, then `autolab init --hill OWNER/HILL --name inference-watch` to create a stopped climb. Confirm version, visibility, node, stop policy and caps before start. An authenticated run needs an attached machine; the frozen hill is CPU-only. Record the report’s tree hash, submit hash, verdict and metrics. | Isolated CLI `0.1.0.post1795037379` verified; whoami reported signed out. Hosted credits and effective cost unverified. `--max-cost` covers agent spend; rented compute is separately priced. Local replay requires no agent or rental. Default authorization for new external charges is $0. [Install and quickstart](https://docs.autolab.ai/quickstart/) · [Cap and stop policy](https://docs.autolab.ai/guides/settings/) · [CLI](https://docs.autolab.ai/reference/cli/) |
| Egbe connection | Owner selects the correct business in Dashboard and completes “Connect your agent” in the repo. Use its ignored `.env.local` and installed quickstart helper. Run the private three resource reads below. Record business identity, remaining units, reset/expiry, hosting allowance and periodic fees privately. | Public catalog and analytics contract reachable; no installed project helper found in the hack checkout. Event credit amount, balance and permissions unknown. The Wallet can incur fees beyond its displayed balance. [Quickstart](https://docs.egbe.ai/start/quickstart/) · [Access and costs](https://docs.egbe.ai/reference/access-and-costs/) |
| Egbe hosting and analytics | After authorization to publish the limited reader surface, obtain its real host. Register that hostname with `POST /analytics`; fetch `GET /analytics/{app}/tracking` and use the returned snippet plus the required recorder setup and privacy handling in the capability instructions. Read `GET /analytics/events` for the public event schema. Use the hosting runtime’s server-side identity for supported app calls. Keep the agent project key out of browser code and hosting settings. | Proposed account-dependent costs are hosting fees + metered operations + any required retention/storage. Verify all three before enabling; no ads, emails or top-ups in this plan. [Analytics instructions](https://os.hz.egbe.dev/analytics/skill) · [Hosting](https://docs.egbe.ai/infrastructure/hosting-and-data/) |
| Claude Code and other APIs | Use the owner’s existing Claude Code access. The service uses a deterministic scoring and scheduling policy. No separate Anthropic, OpenAI, Hugging Face, Stripe, ads or GPU credential is required for the shipped slice. | Claude Code usage follows the owner’s plan or billing method. No runtime LLM API cost in the local worker. |

From the connected project root, substitute the actual helper path reported by Egbe. Use the name of the agent making the call. Inspect account details privately; the branch script prints success markers only.

```sh
export OS_AGENT_NAME="Claude Code"
./scripts/check_egbe_watch.sh "/absolute/path/to/installed/egbe-curl.py"
# For owner-private review, use the installed helper for each resource:
# GET /me/limits    GET /wallet/balance    GET /wallet/policy
```

**Analytics readback.** Identify the deployed site via `GET /analytics`. Read `/analytics/{app}/events` and `/analytics/{app}/event-data` for the same complete window, inclusive UTC milliseconds, pages starting at 1, max 500. Join event `id` to property `eventId`. Check counts and dedupe; a partial page sequence leaves the metric unknown. Use the included workload and opaque receipt metadata to reconcile with the durable queue. Follow the sponsor instructions on permissible properties and retention. If the hosted collector drops a property, use a first-party join inside the product and expose aggregate counts. Never infer a customer from a smoke request. [Egbe OpenAPI](https://os.hz.egbe.dev/openapi.json) · [Session evidence collector](https://os.hz.egbe.dev/skills/egbe-os-analytics/resources/references/session-funnel.md)

**Budget arithmetic.** Per day = fixed project/hosting fees allocated per day + metered events × account unit price + agent charges + compute hours × current rate. Per delivered brief = that total / confirmed deliveries; with no deliveries the figure is undefined. Unknown fee, entitlement or upper bound blocks a new metered action. Local work continues. Approval should name the project, allowed actions, maximum total, expiry and stop rule. Keep Egbe funds and Autolab credits as separate balances. Free participant credits do not establish free indefinitely recurring hosting.

## Verification, demo and handback

1. Open the reader page, request math and code in a rehearsal store, and close the browser. Show the scheduled worker’s independently timestamped decisions and a saved receipt on reopen. Restart midway, replay the same receipt and prove one brief and one charge reservation. Cross-workload receipt reuse must fail.
2. Show two changed observations, their actions and a subsequent measurement. Multiple eligible code requests → allocate code slot → deliver matching briefs. Delivery backlog with missing activations → retry/reconcile delivery → inspect the next complete receipt window. For the contrasting path, inject a labeled rehearsal event or consume a genuine change. Show whether the action helped, failed, or needs more data.
3. Show the local Autolab report for the accepted target and failed draft, plus the Egbe contract/replay or a verified live receipt. Show distinct labels for fresh ML measurement, replayed local evidence, sponsor-hosted scoring, customer observation and synthetic rehearsal. Do not claim the 50% fixture stage rate as traction.
4. Simulate a crash after claiming a job and an external timeout after submission; verify one recovery, one queued follow-up and zero duplicate charges. Test an unknown cost, a stale catalog, missing auth, empty queue, invalid workload, replay exclusion, and degraded analytics.
5. In the runbook record the command that ran, branch and commit, approved budget, last and next tick in Pacific time, host availability, evidence links/IDs, builds/tests, first delivery, limitations and stop command. Record `local verified`, `sponsor authenticated`, `publicly reachable`, and `post-demo runner active` independently. A fixture is sufficient for a rehearsal, never for customer or live integration status.

**90-second pitch.** “Should I turn draft decoding on for my code workload? On this Mac, one setting reproduced only one of eight outputs, twice, and was slower. Here are the run IDs. I’ll request two code briefs and a math brief, then close this tab. The worker picks code first because two readers are waiting, delivers both, and serves math next. Here is what Autolab rejected, what Egbe observed, and when the system will check again.” Use “rehearsal requests” when presenting fixture traffic. If the owner has authorized the ongoing runner, show a check that actually occurred after the demo browser closed.

**Codex requests when blocked.** Put new dependencies in `docs/MOBIUS_LAB_CODEX_REQUESTS.md` in a dated Inference Watch section. Each row needs provider, scope, status, exact owner gesture, official pricing source, estimated maximum, verification command, redacted evidence and next review time. Highest priority is the Egbe business connection and consent to publish the narrow reader surface. Hosted Autolab login is second because the hill already runs locally. Keep implementing all independent steps while the owner handles sign-in. Stop after presenting what runs, what has live evidence, and what still requires an account decision.
