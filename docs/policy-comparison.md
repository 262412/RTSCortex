# Policy Comparison v0.2

Policy Comparison evaluates advisory policies against the same immutable historical
observations. It is shadow-only: proposals are validated and scored, but they are never
sent to Runtime, Bridge, PySC2, or StarCraft II.

## Corpus

The checked-in Protoss and Zerg corpora each contain 48 real protocol v1.1 observations:
eight each for
`early`, `technology`, `production`, `combat`, `blocked`, and `in_progress`. Blocked and
in-progress fixtures retain their underlying phase tag, deterministic goal evidence,
source journal hash, state fingerprint, and successful HIMA-compatible actions from the
previous 60 game seconds.

Verify the materialized corpus without requiring its original journals:

```bash
uv run rtscortex policy-corpus verify \
  benchmarks/policy/protoss_v0_2/manifest.yaml
uv run rtscortex policy-corpus verify \
  benchmarks/policy/zerg_v0_3/manifest.yaml
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
Corpus configs and manifests carry an explicit `race`; manifests created before that field
default to Protoss. Phase detection, goal progress, in-progress effects, and HIMA
`previous_action` projection all come from the active `RaceProfile`. Economy-only actions such
as `Train_Drone` do not by themselves turn a Zerg opening into a production fixture, while a
completed army prerequisite such as `SpawningPool` makes resource-blocked production
representable.

The Terran source pool is recorded in
`configs/policy/corpus_sources_terran_v0_3.yaml`, but it intentionally has no checked-in
manifest yet: the available real journals do not satisfy the two-fixture blocked-production
and blocked-combat phase quotas. The strict builder reports this gap instead of fabricating
states or reducing the quota.

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

The installed Zerg a/b/c checkpoints can be evaluated, one process at a time, with:

```bash
uv run rtscortex policy-compare \
  --config configs/policy/comparison_zerg_v0_3.yaml
```

The first Zerg v0.3 baseline completed all `144/144` specialist-fixture evaluations with
no candidate failure and no illegal Runtime frontier. Classification conservation was 100%.

| Candidate | Parse validity | Mapping coverage | Legal frontiers | Deferred frontiers | Goal advancing |
|---|---:|---:|---:|---:|---:|
| Zerg-a | 100.0% | 94.8% | 11/48 | 37/48 | 0/5 |
| Zerg-b | 99.9% | 96.0% | 8/48 | 40/48 | 0/5 |
| Zerg-c | 100.0% | 96.9% | 8/48 | 40/48 | 1/5 |

This proves integration viability, not strong strategic quality. The largest unsupported
families were Zergling speed, Mutalisk/Spire, Roach speed, Ravager, Overseer, and level-one
combat upgrades. Those gaps should be closed before treating the 27-match suite as a useful
Race Brain quality measurement. Zerg-a currently exposes the most immediately legal
frontiers; Zerg-c provides the broadest mapping coverage and the only goal-advancing frontier
in this corpus.

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

The current live Protoss macro slice maps Zealot, Stalker, Adept, Void Ray, Oracle, Phoenix,
Pylon, Gateway, Cybernetics Core, Assimilator, Nexus, Stargate, Shield Battery, and Warp Gate
research. Stargate placement requires a completed Cybernetics Core, power, 150 minerals, and
150 vespene. Shield Battery requires the completed Core, a powered 2-by-2 placement, and 100
minerals. Adept, Void Ray, Oracle, and Phoenix production use their real resource, supply,
prerequisite, and idle-production-source checks. The melee profile retains the Adept and Void
Ray teams plus both distinct upstream Oracle and Phoenix teams. The new air-specialist teams
are movement-only in this first slice; generic attacks and special abilities remain disabled
until actor-specific target and effect semantics are available.

The checked-in Protoss-a output regression keeps the existing 48-state v0.2 corpus and its
manifest unchanged. Re-scoring those 1,650 effective macro actions with the expanded mapping
must produce 687 mapped actions: 582 future, 91 deferred, and 14 legal now. The remaining
actions are 961 unsupported and 2 parse errors, with no illegal or obsolete actions. This is a
mapping characterization baseline, not a new model inference or gameplay score.

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
