# Policy Comparison v0.2

Policy Comparison evaluates advisory policies against the same immutable historical
observations. It is shadow-only: proposals are validated and scored, but they are never
sent to Runtime, Bridge, PySC2, or StarCraft II.

## Corpus

The checked-in Protoss corpus contains 48 real protocol v1.1 observations: eight each for
`early`, `technology`, `production`, `combat`, `blocked`, and `in_progress`. Blocked and
in-progress fixtures retain their underlying phase tag, deterministic goal evidence,
source journal hash, state fingerprint, and successful HIMA-compatible actions from the
previous 60 game seconds.

Verify the materialized corpus without requiring its original journals:

```bash
uv run rtscortex policy-corpus verify \
  benchmarks/policy/protoss_v0_2/manifest.yaml
```

On the compute-center host, also verify every source journal:

```bash
uv run rtscortex policy-corpus verify \
  benchmarks/policy/protoss_v0_2/manifest.yaml \
  --verify-sources
```

Rebuild it from the configured real journals with:

```bash
uv run rtscortex policy-corpus build \
  --config configs/policy/corpus_sources_v0_2.yaml \
  --output-dir benchmarks/policy/protoss_v0_2
```

The builder fails rather than using synthetic states when any stratum, seed, episode,
game-loop spacing, fingerprint, or blocked/in-progress phase quota cannot be met.

## Offline workflow smoke

The offline configuration exercises corpus loading, candidate accounting, reporting, and
artifact generation without calling Qwen, loading HIMA, importing Transformers, or making
network requests:

```bash
uv run rtscortex policy-compare \
  --config configs/policy/comparison_v0_2.offline.yaml
```

Its expected candidate states are:

- Qwen: `skipped`, because it is disabled;
- HIMA Protoss-a/b/c: `unavailable`, because no local paths or license acknowledgement are
  configured;
- HierNet-SC2: `unavailable: adapter_not_implemented`.

Use `configs/policy/comparison_v0_2.yaml` to evaluate the current OpenAI-compatible Qwen
endpoint. This still does not download or load HIMA weights.

## Local HIMA boundary

HIMA is pinned to upstream revision
`6b2a4084f9d6c0d7739aedb860685cf9dfd90d35`. Each specialist receives only the official
five-field JSON projection: supply used, supply capacity, completed unit/building counts,
completed research, and recent successful actions.

RTSCortex never downloads HIMA weights. To enable a specialist later, its configuration
must provide:

- an absolute local model path;
- the matching pinned model ID and exact Hugging Face cache snapshot revision;
- a Python environment containing Torch, Transformers, and RTSCortex dependencies;
- `allow_unlicensed_weights: true`, after the operator explicitly accepts the risk of the
  checkpoints not declaring a standard license identifier.

The orchestrator starts one subprocess per candidate, evaluates all 48 fixtures, and lets
that process exit before starting the next model. This prevents Protoss-a/b/c from residing
on the GPU simultaneously. The worker forces Hugging Face and Transformers offline modes
and uses `local_files_only=True`.

## Classification semantics

Every discovered logical action has exactly one outcome:

- `parse_error`: malformed, unknown, truncated, or otherwise unparseable output;
- `unsupported_by_runtime`: a valid HIMA action outside RTSCortex's current responsibility;
- `mapped_future`: a valid mapped action after the current runtime frontier;
- `mapped_legal_now`: the frontier maps to a candidate accepted by Runtime validation;
- `mapped_deferred`: mapped, but temporarily unavailable or blocked by current state;
- `illegal_action`: mapped and currently evaluable, but deterministically rejected;
- `obsolete`: its effect is already satisfied.

The sequence frontier is the first model action. The runtime frontier skips actions such as
`Probe`, which RTSCortex currently manages automatically. It also distinguishes two kinds of
mapped blocker:

- a soft blocker, such as a missing camera candidate or temporarily unavailable actor, does
  not hide the next mapped action that Runtime can validate now;
- a hard blocker, such as insufficient resources, insufficient supply, or an incomplete tech
  prerequisite, preserves the HIMA sequence and becomes the runtime frontier.

The first legal, illegal, or hard-deferred action is the unique runtime frontier. If every
mapped action is soft-deferred, the earliest one is the frontier. Unsupported, deferred,
future, obsolete, and parse-error actions never inflate the illegal-action rate. Reports
retain both logical-step counts and repeat-weighted effective-action counts.

The current live Protoss macro slice maps Zealot, Stalker, Adept, Void Ray, Pylon, Gateway,
Cybernetics Core, Assimilator, Nexus, Stargate, and Warp Gate research. Stargate placement
requires a completed Cybernetics Core, power, 150 minerals, and 150 vespene. Adept and Void
Ray production use their real resource, supply, prerequisite, and idle-production-source
checks. The melee profile also retains one Adept and one Void Ray combat team so newly
produced units remain controllable.

## Artifacts

Each run creates:

```text
policy-comparison-<timestamp>/
├── comparison.json
├── report.md
├── config.snapshot.yaml
├── corpus.snapshot.yaml
└── candidates/<candidate-id>/
    ├── records.jsonl
    └── provenance.json
```

`comparison.json` is lossless. `report.md` summarizes availability, parser validity,
mapping coverage, frontier results, goal progress, control-action violations, latency, and
per-stratum outcomes. Candidate provenance records the model identity, revision, HIMA
adapter/parser/vocabulary versions, execution backend, and the fact that downloads were
disabled.
