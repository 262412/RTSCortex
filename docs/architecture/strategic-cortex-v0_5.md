# Strategic Cortex v0.5 architecture

## Runtime pipeline

```text
ObservationEnvelope
  -> SituationProvider
  -> active RaceProfile and HIMA a/b/c Race Brain
  -> Economy / Technology / Production / Defense / Offense / FocusFire / Retreat
  -> PlaybookIntentGuard
  -> IntentArbiter
  -> CandidateCompiler
  -> PlaybookCandidateGuard
  -> Fast Executor
  -> ProgressGuard and Validator
  -> LLM-PySC2 Bridge
  -> PySC2
  -> EffectVerifier and post-game review
  -> CortexPlaybook
```

Only the active player's RaceProfile and HIMA ensemble are loaded. The seven role agents
share that ensemble; they do not each load another language model. Role agents emit typed
`StrategicIntent` values and cannot dispatch `ActionCommand` values directly.

The Fast Executor remains deterministic. It selects only from validated candidates and does
not perform strategic reasoning.

## Safety and activation boundaries

- The control wire protocol remains v1.1. Situation, Intent, Playbook and RaceProfile values
  are internal Cortex v2 schemas recorded as additive events.
- The default Intent Arbiter mode is `shadow`. It emits decisions and diffs without changing
  the live ActionBatch.
- The default executable Playbook mode is `shadow`. Legacy lessons migrate as advisory and
  cannot block an action.
- `active` must be configured explicitly after paired-seed gates pass.
- HIMA receives its official five-field observation. Playbook text is never injected into the
  HIMA prompt; typed guards are applied after proposal parsing.
- A live Cortex configuration must name one concrete player race, and the HIMA checkpoint
  race must match it.

## Current race capability matrix

| Capability | Protoss | Terran | Zerg |
|---|---|---|---|
| Fixed HIMA a/b/c vocabulary, parser and adapter | Ready | Ready | Ready |
| Local-only a/b/c checkpoint validation | Ready | Ready | Ready |
| Offline three-member inference smoke | Ready | Ready | Ready |
| RaceProfile macro contract | Ready | Ready | Ready |
| Runtime action mapping | Ready | Ready for current Simple64 frontier | Ready for current Simple64 frontier |
| Live LLM-PySC2 Worker | Ready | Ready | Ready |
| Effect verification | Build, production and move | Build, production, add-on and move | Build, production, morph, inject and move |
| Live 48-state corpus | Protoss v0.2 | Terran v0.3 | Zerg v0.3 |
| Seeds 0/1/2 engineering regression | Ready | Ready | Ready |

The pinned LLM-PySC2 environment remains unchanged, while reviewed Bridge adapters provide
race-specific observation, actor routing and action/effect semantics. All three races use the
same Cortex runtime. Current explicit gaps are Terran research/morph verification, chained
Zerg creep spread, and the 27-match tactical-quality suite.

The installed specialist checkpoints are pinned and loaded with `local_files_only=True`:

| Checkpoint | Snapshot revision |
|---|---|
| SNUMPR/Terran-a | `192e1117a64417cd70f7e663008e61f6c5ced9b8` |
| SNUMPR/Terran-b | `f9eec984998e727598336813b0b63b7dad3012c8` |
| SNUMPR/Terran-c | `32df87695ea1d3b69a8d1592b14857572d540012` |
| SNUMPR/Zerg-a | `c2bb6b08f531cd96d46c83e004c1312066855a46` |
| SNUMPR/Zerg-b | `8b74c24921ee921445b27b1e7ce13523cbde9c0a` |
| SNUMPR/Zerg-c | `79322205ba70ccd56a72645164178aa027d2fbf6` |

All six snapshots passed Hub checksum verification, isolated GPU load health, and one real
three-member offline inference smoke. Their model cards do not declare a weight license; use
requires the explicit `allow_unlicensed_weights` acknowledgement already accepted for this
environment.

## Situation Intelligence v2

The deterministic provider emits typed facts for phase, economy, army readiness, visible
force composition, confirmed or inferred technology, bases, production, threats, spatial
control and scouting. Every inference carries a source and confidence. Unobserved values are
represented as unknown rather than filled with model guesses.

`model_shadow` and `model_active` are provider modes. A learned provider may become active
only after it matches or exceeds the deterministic baseline on at least 300 labelled states
and produces no severe hallucinations.

## Strategic arbitration

The Intent Arbiter chooses a feasible subset of at most seven role intents. It resolves:

- actor and producer conflicts;
- mineral, vespene and supply reservations;
- dependencies and mutually exclusive objectives;
- current commitments and switching costs;
- emergency retreat, defense and reflex preemption.

Every input intent receives exactly one `selected`, `deferred`, `rejected` or `preempted`
decision. Stable identifiers break ties, so journal replay is deterministic.

## Executable CortexPlaybook v2

Rules use typed, whitelisted conditions and one of `require`, `forbid`, `prefer` or `avoid`.
They never execute Python, SQL or model-generated code. Rule applications are capped at eight
hard and eight soft matches per decision.

Legacy lessons are preserved as advisory. New evidence is merged by canonical
condition/effect rather than by text, and repeated evidence from the same run does not inflate
support. Candidate-to-soft promotion requires independent runs. Hard promotion additionally
requires the configured seed, revision, false-block and paired A/B gates. Independent
contradictions reduce confidence and eventually suspend or retire a rule.

## Event lineage

The Console and report consume additive events including:

```text
race_profile_activated
situation_assessed
tactical_policy_shadow
role_intent_emitted
intent_arbitrated
intent_arbiter_shadow_diff
playbook_rule_applied
playbook_rule_updated
command_lineage
```

For dispatched commands, lineage links the active race, Race Brain plan, role, strategic
intent, arbitration decision, Playbook rules, candidate and terminal execution result.

## Activation sequence

1. Run the Protoss a/b/c ensemble with Arbiter and Playbook in shadow mode on seeds 0, 1 and
   2; verify deterministic replay, reservations, ownership and no command-success regression.
2. Promote the Arbiter to active only after the shadow engineering gates pass.
3. Run paired Playbook on/off seeds; promote only the rules that satisfy the published gates.
4. Completed: Terran Worker, add-on effects, smoke and seed regression.
5. Completed: Zerg larva, inject, creep/morph provenance, seed regression, and 48-state corpus.
6. Completed: Terran blocked-production and blocked-combat coverage and its verified
   48-state corpus.
7. Complete the remaining race-effect gaps, then run the 27-match tactical-quality suite.

This order intentionally keeps “model can produce a proposal” separate from “the race is safe
to execute in PySC2.”
