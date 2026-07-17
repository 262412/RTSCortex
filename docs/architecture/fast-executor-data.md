# Fast Executor data foundation

RTSCortex exports the deterministic executor boundary as a small, auditable corpus. The
export is intended for policy comparison and a future tiny motor-policy model; it does not
train or download a model and it cannot dispatch an SC2 action.

## Data boundary

Each valid `executor_selection` produces one sample by joining the following protocol 1.1
events:

```text
observation + intent_emitted + candidate_set_built + executor_selection
                                          |
                                          + command_lineage
                                                |
                                                + command_lifecycle / execution
```

The exporter first joins runtime events with their original IDs, then replaces every
runtime observation/candidate/selection/intent/command ID with a deterministic corpus-local
ID. Those local IDs are derived only from the compact observation, source identity, semantic
action names, deterministic rank features, and the candidate's local ordinal. Actor names,
arguments, tags, coordinates, and hashes computed from those values never cross the export
boundary. The label is either an exact corpus-local `candidate_id` from the saved candidate
domain or an explicit abstention. A linked terminal execution outcome is optional because selection can
be observed before dispatch or before the episode supplies a terminal report. Invalid joins
are excluded and counted by reason; they are never silently converted to a label.

The compact observation contains economy values, aggregate unit/structure/enemy/production
counts, upgrades, available action names, and the alert count. It deliberately excludes:

- RGB frames and image references;
- observation prose and model prompts;
- candidate arguments, actor instances, unit tags, and coordinates;
- runtime observation fingerprints and runtime candidate, selection, intent, command,
  macro-plan, and situation-assessment IDs;
- raw journal byte hashes, because they also hash the redacted executable values;
- raw failure text (only structured status, stage, and failure code are retained);
- credentials and provider configuration.

Candidate rows retain only a corpus-local ID, action name, and deterministic rank features.
The ordered semantic `intent_action_names` are retained so that an empty candidate domain is
still interpretable. This is enough to reproduce the current candidate-ranking policy, but not enough
to bypass the live CandidateCompiler, Validator, Bridge, or effect verifiers.

## Build and verify

```bash
uv run rtscortex executor-corpus build \
  /path/to/run-a /path/to/run-b/events.jsonl \
  --output-dir /path/to/executor-corpus

uv run rtscortex executor-corpus verify \
  /path/to/executor-corpus/manifest.json \
  --verify-sources
```

The output contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`.
Splits use a stable SHA-256 assignment over `(split seed, run ID, episode ID)`, so an episode
can never leak across splits. Samples whose compact semantic state would otherwise appear in
more than one split are deterministically retained only in the split owned by the canonical
episode and reported as `cross_split_semantic_duplicate` exclusions. This avoids joining whole
episodes through their shared opening state while keeping evaluation states disjoint. The
manifest records source fingerprints, artifact hashes, schema and builder versions, exclusions,
duplicate counts, conservation checks, and class/action/outcome distributions. Duplicate-state
checks use a separate semantic feature hash, because the corpus-local observation fingerprint
intentionally contains run/episode/step identity.

Source provenance is a hash of an allow-listed safe event projection, not a byte hash of the
raw JSONL. The manifest keeps only a path relative to the corpus directory for optional local
source verification; it does not persist the user's absolute storage path. `--verify-sources`
therefore detects changes to compact observations, semantic
intents/candidates/selections, lifecycle states, and structured outcomes, while deliberately
ignoring changes confined to redacted IDs, tags, coordinates, actors, arguments, prose, and
images. Artifact SHA-256 values still verify the exported corpus bytes exactly.

## Saved-candidate benchmark

```bash
uv run rtscortex executor-benchmark \
  /path/to/executor-corpus/manifest.json \
  --repetitions 100
```

The default evaluates the `test` split; use `--split validation`, `--split train`, or
`--split all` explicitly for other views. Primary agreement is computed over selected labels;
overall agreement and abstention matches are reported separately. The command also reports CPU
p50/p95/p99 for ranking the saved candidates. It is not
an end-to-end Runtime latency benchmark: compact samples cannot reconstruct an
`ObservationEnvelope`, executable arguments, validation, PySC2 translation, or effect
verification. Live executor latency remains an independent runtime acceptance metric.
