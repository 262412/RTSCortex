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

The command writes `<run-directory>/timeline.md`. It summarizes each observation tick,
typed model output, accepted plan, selected and rejected commands, reflex preemptions,
execution feedback, latency, token use, and terminal result while preserving the JSONL
event order. It intentionally omits the verbose raw text observation from Markdown; the
complete source remains in `events.jsonl`. Running the command again deterministically
replaces the derived timeline and does not modify the journal.
