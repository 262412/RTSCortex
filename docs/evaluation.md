# Evaluation artifacts and metrics

`rtscortex eval` runs the four v0.1 offline baselines for every configured seed and
writes a self-contained artifact directory:

- `config.yaml`: the fully resolved experiment configuration.
- `provenance.json`: code and upstream commits, dirty-state flags, seeds, model,
  prompt versions and hashes, adapter version, Python version, and platform.
- `episodes.jsonl`: one result per variant and seed with derived episode metrics.
- `summary.json`: per-variant aggregate metrics and provenance.
- `report.md`: a compact human-readable comparison.
- `runs/`: the SQLite state, append-only event journal, and resolved
  config/provenance for each variant and seed.

The runner rejects a non-empty output directory instead of appending to an older
SQLite database or event journal. Use a new directory for every suite.

The report separates decision activity, meaningful outcomes, the build funnel, and failure
taxonomy. The first build-funnel stage counts every raw `proposed_actions` build emitted by
the Planning model, before ActionModule limits, Runtime validation, or actor arbitration.
Semantic NoOps are excluded from gameplay rates; transport NoOps are counted separately. It
reports success over all meaningful commands and over completed commands,
cancelled/unconfirmed backlog, terminal-report coverage, unexpected terminal reports, and
explicit stage/code coverage. Execution-as-dispatch reconstruction is used only for protocol
1.0 journals that contain no lifecycle events; protocol 1.1 cancellation reports for commands
that never crossed the dispatch boundary remain correctly classified as non-dispatched.
Historical protocol 1.0 execution rate remains visible only as a deprecated compatibility
metric.

The safety and attribution section audits the same raw Planning outputs for Builder-owned or
friendly-target `Attack_Unit` proposals. It separates unsafe proposals stopped before
dispatch from those actually dispatched, while the dispatched-command counters independently
cover Planner and Reflex commands. The hard gates therefore fail on unsafe model output even
when Runtime validation successfully blocks it, and also fail on any unsafe command that
crosses the dispatch boundary. Proposal-audit coverage, Planner NoOps, generic translation
failures, unattributed primitives, upstream placement rejections, candidate-external
dispatches, and orchestration function `573` being mistaken for a terminal primitive are all
explicit. Plan-acceptance gap percentiles are shown only when at least one gap sample exists;
otherwise the report says that the sample is insufficient rather than printing a misleading
zero.

Action success is calculated from execution reports. Illegal actions are commands
rejected by the runtime validator before execution, so their rate uses all executed
and rejected candidate actions as its denominator. Planner, reflex, and full-tick
latencies are calculated directly from runtime event samples; suite p50/p95 values
pool those samples across seeds.

The plan revision rate is the number of semantically changed accepted plans divided
by the number of opportunities after each episode's initial plan. Token cost uses the
provider's configured prompt and completion prices per million tokens. Both prices
default to zero, so cost remains explicit without assuming a vendor price.

## Per-run timeline

Generate a readable account of any live or Mock run directly from its append-only
event journal:

```bash
uv run rtscortex report ~/scratch/outputs/RTSCortex/<run-directory>
```

The command writes two deterministic derived artifacts:

- `<run-directory>/timeline.md` summarizes each observation tick, typed model output,
  accepted plan, selected and rejected commands, reflex preemptions, execution feedback,
  latency, token use, terminal backlog, and the hard acceptance gates.
- `<run-directory>/summary.json` groups results by run and episode and records the optional
  `EpisodeResult`, episode or incomplete execution metrics, classification conservation,
  dispatched-command terminal-report coverage, and machine-readable gate values and
  pass/fail results.

The Markdown preserves JSONL event order and intentionally omits the verbose raw text
observation; the complete source remains in `events.jsonl`. Running the command again
deterministically replaces both derived reports and does not modify the journal. The
`rtscortex run` command performs the same report step automatically after normal completion
and after failures that have already recorded events. Report-generation errors are warnings
and never replace the original run result or error.
