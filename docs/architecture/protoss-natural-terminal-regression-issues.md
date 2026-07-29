# Protoss natural-terminal regression issue register

Status date: 2026-07-28

This register contains only issues that remain open after the latest natural-terminal
validation:

```text
run set:
  /mnt/scratch/users/tbczhang/outputs/RTSCortex/
  protoss-evolving-seed2-natural-terminal-20260728T0224Z

run directory:
  /mnt/scratch/users/tbczhang/outputs/RTSCortex/
  cortex-20260728T012349790480Z-732880c5

source revision:
  ce9d095e0f15c323fdb689191ca1f0f46c392255
```

The run used HIMA Protoss a/b/c, active Strategic Intent Arbiter, evolving
CortexPlaybook, `Simple64`, Protoss versus VeryEasy Zerg, PySC2 Raw Actions,
`game_steps_per_episode: 0`, `step_mul: 1` and
`simulation_speed_multiplier: 1.0`.

| Seed | Terminal | Game loops | Wall time | Meaningful success | Build | Production |
|---|---|---:|---:|---:|---:|---:|
| 2 | victory | 27,294 | 10 h 27 min 52 s | 2,180/2,285 = 95.4% | 34/44 = 77.3% | 79/79 = 100% |

The Slurm job completed normally with exit code `0:0`. There were no runtime
crashes, HTTP 422 responses, Bridge integrity errors, duplicate dispatches,
friendly-target attacks, unattributed primitives or observation-gap watchdog
triggers. Thirty-two hard acceptance gates passed, two build-effect gates failed
and one gate was not applicable.

The aggregate success rate is not yet a reliable tactical-quality metric:
`Attack_Unit` for `CombatGroup3/VoidRay-1` contributed 1,504 reports, and the
same target received up to 101 separately tracked successful attack commands.

## 2026-07-28 review remediation

The follow-up review of revision `a2886c5` found four behavior blockers and
several partially closed lifecycle contracts. The code corrections below are
implemented together so that an identity, not an actor name or approximate
coordinate, crosses each asynchronous boundary. They require a fresh live smoke
before this register may claim runtime acceptance.

### Placement and building lifecycle

- **Root cause:** placement exclusion compared center-point radii, reservations
  were created only at final dispatch, and ordinary confirmed structures were
  released without an authoritative occupied footprint.
- **Correction:** every build reservation now records structure type, exact
  width/height, occupied build-grid cells, original and emitted world target,
  state, episode, revision and expiry. Candidate-stage screen metadata carries
  an immutable placement candidate ID through Router and RawExecutor. Conflict
  checks use cell intersection across structure types; observations maintain an
  occupied-structure ledger; confirmed reservations remain occupied until the
  observed structure takes ownership. Builder leases expire and all other RAW
  orders reject leased tags.
- **Regression evidence:** `test_cross_structure_footprints_cannot_overlap`,
  `test_actor_failure_does_not_quarantine_placement`, placement provenance and
  build-effect tests.

### Expansion goal versus scouting epoch

- **Root cause:** `expansion_candidates_exhausted` terminalized both the active
  commitment and its durable desired-base-count goal. A later scout generation
  therefore could not reopen the goal.
- **Correction:** candidate exhaustion now closes only the current commitment,
  records the exhausted epoch, decrements the bounded global retry budget and
  moves the goal to `waiting_for_candidates`. A newer scout generation returns
  it to `active`; only satisfaction, strategic cancellation, episode end or
  genuine global retry exhaustion is terminal.
- **Regression evidence:** `test_expansion_goal_reopens_on_new_candidate_epoch`
  and commitment recovery tests.

### Combat, retreat and Defense lineage

- **Root cause:** target damage could confirm an Attack without evidence that
  the exact actor held the target order; tactical and Defense reports could fall
  back to actor/action matching; threat signatures included unstable visible
  tags; successful Defense responses entered an unbounded `holding` state.
- **Correction:** combat success requires PySC2 acceptance, an exact actor RAW
  order observed against the target, and target damage or valid removal. An
  overwritten order produces `combat_order_replaced`, allowing the tactical
  agent to reissue. Dispatch binds command, operation and attempt to the current
  engagement/retreat/Defense state, and stale reports are ignored. Retreat
  signatures use stable threat class/domain rather than tag membership. Defense
  holding expires and is re-evaluated under sustained threat until inventory
  saturation guards stop further production.
- **Regression evidence:** `test_unrelated_damage_does_not_confirm_attack`,
  `test_overwritten_attack_order_can_be_reissued`,
  `test_stale_report_cannot_mutate_new_retreat_commitment` and
  `test_defense_holding_expires_under_sustained_threat`.

### Persistence, HIMA and experiment gates

- **Root cause:** EventStore used an unbounded writer queue and synchronous
  subscriber callbacks, JSONL had no repair path, HIMA globally merged repeated
  actions across intervening dependencies, and Playbook gates treated absent
  shadow evidence as success while measuring benefit from sequential carry.
- **Correction:** the durable queue and subscriber queues are bounded;
  subscribers run on isolated workers; overload applies explicit durable
  backpressure; queue depth, writer lag, journal bytes and append latency are
  observable. Startup reconciles a malformed or lagging JSONL mirror from
  canonical SQLite. HIMA compacts only contiguous identical actions, preserves
  interleaving and resolves transitive prerequisites. Playbook benefit is
  measured on independent pairs, repeated errors use canonical consequence
  signatures, natural terminal requires victory/defeat/draw, and zero shadow
  states fails the hard-rule false-block gate.
- **Regression evidence:** `test_hima_compaction_preserves_interleaved_order`,
  `test_event_writer_queue_is_bounded`, JSONL reconciliation tests and
  `test_false_block_gate_cannot_pass_without_shadow_states`.

The short persistence smoke is:

```bash
uv run python scripts/profile_event_store.py
```

It reports `loops_per_second`, `events_per_loop`, `bytes_per_loop`,
`queue_peak`, `queue_capacity`, `writer_lag_ms_max` and append latency. These
metrics validate the persistence subsystem; the next SC2 smoke must still
measure end-to-end live loop speed before Frozen/Evolving acceptance.

Local 2,000-loop / 12,000-event smoke after this correction:

```text
loops_per_second: 7701.33
events_per_loop: 6.0
bytes_per_loop: 1367.79
queue_peak/capacity: 8192/8192
writer_lag_ms_max: 165.07
append_latency_ms_mean: 0.0097
```

The deliberate saturation proves that memory remains bounded and backpressure
engages. It is not a substitute for the required SC2 end-to-end live-speed
measurement.

## 2026-07-28 follow-up review closure

The review of `47480da` identified three remaining behavior blockers, three
recovery/performance/analysis gaps and two hidden cross-race boundaries. The
following corrections close the code-level findings. The independent
Frozen/Evolving twelve-run acceptance remains a separate experiment under
SCX-PT-039; it is not implied by this code-review closure.

### Dispatch-owned tactical and strategic commitments

- **Root cause:** Tactical and Defense agents mutated engagement, retreat and
  holding state while merely proposing an Intent. The strategic agenda was
  likewise committed after arbitration but before validation and dispatch.
  Additionally, the Arbiter selected only one Intent per role even when actor
  scopes were disjoint.
- **Correction:** proposal evaluation is now side-effect free. Engagement,
  retreat, offense navigation and Defense operation state are created only by
  `record_dispatch()` for an accepted command. Strategic arbitration remains
  provisional until the matching command enters `DISPATCHED`, at which point a
  `strategic_agenda_committed` event is stored. Same-role Intents conflict only
  when their actor, producer or objective claims overlap.
- **Evidence:** regression tests cover undispatched FocusFire, Retreat and
  Defense proposals, two disjoint FocusFire groups, and an arbitrated Intent
  rejected before command dispatch.

### Current-order combat attribution

- **Root cause:** `actor_order_bound` represented whether the actor had ever
  held the target order. A later health delta could therefore confirm an old
  command after the actor had moved or received another order. Any targeted
  ability with the expected tag could also look like an attack.
- **Correction:** the historical binding is retained only as diagnostic
  evidence. Success requires the current observable order to have both the
  expected target tag and a pinned SC2 attack ability ID, followed by target
  damage or valid removal in the causal window. Replacement and missing-actor
  timers cannot be bypassed by another actor's damage.
- **Evidence:** tests cover damage after order replacement, non-attack targeted
  abilities and actor disappearance after a prior valid binding.

### Hard HIMA executable horizon

- **Root cause:** `MacroPlan` projected a bounded prefix, but Runtime selected
  remaining work from the full compact HIMA proposal. An ordinal absent from the
  active plan could consequently be compiled with no plan-step lineage while a
  replan was pending.
- **Correction:** Runtime builds an executable ordinal whitelist exclusively
  from the active `MacroPlan`. Exhausting that set freezes the old proposal,
  records `macro_executable_horizon_exhausted` and requests an urgent replan.
  Full HIMA output remains provenance only.
- **Evidence:** an end-to-end Runtime test proves that the first opaque future
  step cannot dispatch before a replacement plan is accepted.

### Checkpoint recovery continuity

- **Root cause:** tail replay restored expansion start/terminal events but not
  reopen or candidate-epoch exhaustion. Attempt ordinals created after the last
  snapshot were not folded back into the operation counters.
- **Correction:** recovery replays the latest expansion goal transition,
  including phase, generation, exhausted epoch, retry budget and cooldown.
  Command lineage persists `attempt_ordinal`, and tail replay advances each
  operation counter to one beyond its maximum durable attempt.
- **Evidence:** restart tests cover reopen, exhaustion and monotonically
  increasing attempt ordinals after a checkpoint.

### Live EventStore performance acceptance

- **Root cause:** the synthetic writer profile proved bounded memory but did not
  measure real SC2 event payloads, writer lag or end-to-end game-loop speed.
- **Correction:** EventStore now exposes writer-lag p95/max and blocked durable
  appends. Runtime emits an
  `event_store_performance` terminal event, and the profiling/report path
  includes event and byte exposure. Durable events do not currently implement
  a sampled-drop lane, so the report states
  `sampled_drop_supported: false` instead of publishing an inert zero counter.
- **Real smoke:** Slurm job `9988137`, HIMA Protoss-a, active Cortex, 508-step
  bounded live smoke, completed with exit code `0:0`.

```text
game_loops_per_second: 15.56
events_per_game_loop: 0.275
bytes_per_game_loop: 866.29
writer_queue_peak/capacity: 7/8192
writer_lag_ms_p95: 420.75
writer_lag_ms_max: 500.53
blocked_append_count: 0
sampled_drop_supported: false
artifact_size_bytes: 1,546,192
```

The live producer did not approach queue saturation and performed no blocked
durable append. Because sampled dropping is unsupported, this run makes no
claim about sampled-event loss. This closes the single-run persistence
performance gate; natural-terminal artifact growth must still be reported by
the paired acceptance experiment.

### Terran add-on footprint ownership

- **Root cause:** Barracks, Factory and Starport candidate checks considered
  add-on clearance, while the persistent world ledger reserved only the main
  structure cells. A later structure could occupy future Tech Lab/Reactor
  space.
- **Correction:** Terran producers use one 5-by-3 world-space footprint ledger:
  the 3-by-3 main body plus the pinned two-column add-on area. Candidate
  generation, validation, reservation and observed occupancy share this exact
  cell set.
- **Evidence:** a placement contract test locks all fifteen cells and verifies
  that another structure cannot overlap the add-on region.

### Streaming, exposure-normalized Playbook analysis

- **Root cause:** experiment analysis materialized complete JSONL journals in
  memory and compared only raw repeated-error totals, biasing runs of different
  duration.
- **Correction:** the analyzer makes one streaming pass over each journal and
  reports repeated errors as a total, per 10,000 game loops and per lineaged
  operation. Pair-level reduction uses pooled game-loop exposure. It also
  reports live event rate, byte rate, writer performance and total artifact
  size.
- **Evidence:** tests use a one-shot iterator to reject a second pass and lock
  both normalized denominators. The all-lineage denominator is named explicitly
  and is not presented as an error-specific eligible exposure.

### Defense-unit saturation

- **Root cause:** static-defense structures had completed/pending/reserved
  limits, but emergency unit production could re-arm after each holding window
  without counting completed units, production queue and dispatched responses
  together.
- **Correction:** every RaceProfile now declares per-unit defensive saturation
  limits. Defense compiles a training Intent only when
  `completed + queued + dispatched response < limit`.
- **Evidence:** a sustained air-threat test verifies that five Phoenixes plus
  one queued Phoenix satisfy the Protoss limit and suppress another emergency
  training proposal.

## 2026-07-28 final pre-acceptance review closure

The review of `0d5d4c6` identified two formal-acceptance blockers, one
FocusFire/Playbook metric contaminant and three semantic or observability
defects. All six are closed at code and deterministic-test level below. This
does not itself claim that the independent Frozen/Evolving multi-seed
experiment under SCX-PT-039 has run.

### Expansion continuation cannot authorize an opaque future Nexus

- **Evidence:** the bounded `MacroPlan` may contain ordinals 0-4 while the full
  HIMA provenance still contains a Nexus at ordinal 5 or later. An expansion
  commitment previously inspected the full proposal, then injected a synthetic
  Nexus into the remaining proposal when a current step was deferred.
- **Impact:** Runtime could execute a macro action outside the finite HIMA
  execution horizon, invalidating strategy attribution and paired evaluation.
- **Root cause:** expansion authorization and ordinary macro authorization used
  different sources: `MacroPlan.steps` for ordinary actions and the complete
  `MacroPolicyProposal` for expansion.
- **Correction:** `_ensure_expansion_commitment()` now intersects the proposal
  with the active plan's executable ordinals. An undispatched commitment is
  cancelled when no current executable townhall step authorizes it. Synthetic
  continuation is permitted only after the expansion command has actually
  entered `DISPATCHED`, and that dispatched bit is checkpointed and recovered.
- **Acceptance evidence:**
  `test_opaque_future_expansion_cannot_dispatch_before_replan`,
  `test_deferred_horizon_step_does_not_unlock_future_nexus` and
  `test_active_dispatched_expansion_can_continue_without_reauthorizing_future_step`.

### Hard-rule false-block accounting is event-time based

- **Evidence:** before/after SQLite snapshots filtered rules by their final
  `active + hard` state. A hard rule suspended or retired during a run lost
  current-run false blocks, while a soft rule promoted to hard imported
  historical counters.
- **Impact:** a twelve-run experiment could falsely pass or fail the hard-rule
  false-block gate even when every game completed successfully.
- **Root cause:** mutable rule-lifecycle snapshots were used as a substitute for
  immutable per-decision evidence.
- **Correction:** each application now records a `playbook_rule_evaluated`
  event containing rule strength/status at evaluation time, shadow decision,
  target, actual terminal outcome and false-block result. The analyzer folds the
  latest record for each evaluation ID directly from the run journal. Active
  hard `would_block` evaluations without observable counterfactual outcomes
  remain unresolved regardless of whether their terminal label is `blocked`,
  `not_selected`, `cancelled`, `unconfirmed`, `satisfied_by_peer` or `pending`.
  Only resolved `would_block` evaluations enter the false-block denominator;
  `would_allow` observations never dilute it.
- **Acceptance evidence:**
  `test_false_blocks_are_preserved_when_hard_rule_becomes_suspended`,
  `test_soft_to_hard_transition_does_not_import_historical_false_blocks`,
  `test_retired_rule_run_delta_uses_event_time_strength` and
  `test_unresolved_active_hard_block_cannot_pass_false_block_gate`.

### One killed target produces one kill and neutral peer terminals

- **Evidence:** multiple exact-bound actors can attack the same target. The
  first command claimed `target_removed`; peers remained pending and later
  became `combat_target_lost` or timeout failures.
- **Impact:** valid FocusFire engagements generated false tactical failures,
  repeated-error signatures and incorrect Playbook lessons.
- **Root cause:** damage evidence was intentionally one-to-one, but target
  removal had no engagement-level terminalization rule for other actors that
  had held the same exact attack order.
- **Correction:** target removal still confirms exactly one command as the kill.
  Other accepted commands in the same engagement whose actors were previously
  exact-bound and had not received a confirmed replacement before target death
  end as `cancelled / engagement_target_eliminated` with
  `confirmation_kind=satisfied_by_peer`. Tactical state treats this as a
  satisfied engagement, the reviewer treats it as inconclusive rather than an
  error, and metrics expose a separate neutral
  `meaningful_satisfied_by_peer` category excluded from successes, failures and
  backlog.
- **Acceptance evidence:**
  `test_target_removal_terminalizes_all_exact_bound_commands_in_engagement` and
  `test_peer_satisfied_attack_is_not_counted_as_failure_or_duplicate_kill`.

### Persistence reports unsupported sampled dropping honestly

- **Evidence:** durable queue saturation blocks or raises; no production path
  identifies and drops sampled telemetry. The old
  `dropped_sampled_event_count` therefore remained zero by construction.
- **Impact:** reports could present an inert zero as proof that lossy telemetry
  had no drops.
- **Root cause:** a planned sampled telemetry lane was represented in the
  metrics contract before it existed.
- **Correction:** the inert counter is removed from `EventStorePerformance`,
  profiler and experiment analyzer. Reports now publish
  `sampled_drop_supported: false`. Subscriber drop accounting remains a
  separate, real metric.
- **Acceptance criterion:** no report may claim sampled-drop losslessness until
  a real sampled lane and increment path exist.

### Attempt ordinals count dispatches, not materializations

- **Evidence:** candidate compilation previously incremented the operation's
  attempt ordinal before ProgressGuard, candidate validation, arbitration and
  final validation.
- **Impact:** a rejected candidate could consume ordinal 0, causing the first
  real dispatch to be labelled attempt 1.
- **Root cause:** attempt identity was bound where a command object was
  materialized rather than at the accepted-dispatch boundary.
- **Correction:** materialized commands carry the stable operation ID but no
  attempt ID. After final validation accepts the command, Runtime binds the next
  ordinal immediately before dispatch persistence and records the bound lineage.
  Lifecycle equality permits only this single immutable enrichment.
- **Acceptance evidence:**
  `test_rejected_materialization_does_not_consume_dispatch_attempt_ordinal`;
  restart recovery continues to advance from the maximum durable dispatched
  ordinal.

### Operation-normalized metrics use an accurate denominator name

- **Evidence:** the analyzer denominator contains every unique operation present
  in command lineage, including operations unrelated to a particular strategic
  consequence.
- **Impact:** the old name `eligible_operation_count` overstated the semantic
  specificity of the exposure and could mislead downstream analysis.
- **Root cause:** no error-type-specific exposure predicate exists in the
  current event schema.
- **Correction:** fields are renamed to `lineaged_operation_count` and
  `repeated_errors_per_lineaged_operation`. The primary paired acceptance gate
  remains the pooled game-loop-normalized rate. A truly eligible denominator
  must wait for explicit per-error exposure predicates.
- **Acceptance evidence:** the analyzer regression includes an unrelated
  lineaged operation and verifies the honest all-lineage denominator.

## 2026-07-28 acceptance-boundary review closure

The final acceptance-boundary review found two orchestration false-pass paths
and one remaining FocusFire attribution edge. They are closed at code and
deterministic-test level below. The twelve-run Frozen/Evolving experiment still
must be rerun; this section does not mark SCX-PT-039 complete.

### Rejected comparison reports now fail the experiment process

- **Evidence:** the analyzer persisted `comparison.json` with
  `accepted=false`, then returned normally. The paired shell runner propagated
  only the twelve SC2 child exit codes.
- **Impact:** all game processes could exit zero while one or more aggregate
  engineering or causal gates failed, causing Slurm and external orchestration
  to record a false pass.
- **Root cause:** the report producer and experiment process used independent
  success contracts. No exit status crossed the report/runner boundary.
- **Correction:** the analyzer writes both JSON and Markdown, then exits zero
  only when every gate is accepted. The paired runner captures that status
  under `set +e`, promotes any analyzer rejection to `overall_status=1`, and
  persists `analysis_exit_code` beside the final experiment exit code.
- **Acceptance evidence:**
  `test_analyzer_cli_exits_nonzero_when_report_rejected`,
  `test_paired_runner_propagates_failed_acceptance_gate`, and shell syntax
  validation.

### Hard false-block accounting covers every unobservable blocking counterfactual

- **Evidence:** only `would_block + actual_outcome=blocked +
  false_block=None` was unresolved. `not_selected`, `cancelled`,
  `unconfirmed`, `satisfied_by_peer` and `pending` remained invisible, while
  resolved `would_allow` evaluations incorrectly increased the denominator.
- **Impact:** many harmless `would_allow` samples could dilute an unobservable
  hard blocking counterfactual and allow the aggregate gate to pass.
- **Root cause:** the analyzer filtered by a downstream outcome label rather
  than first selecting the counterfactual population defined by
  `shadow_decision=would_block`.
- **Correction:** active hard evaluations are first restricted to
  `would_block`. Boolean `false_block` values are the only resolved
  denominator; every `None` is unresolved and blocks acceptance. `would_allow`
  is excluded from all false-block counts.
- **Acceptance evidence:** regressions cover unselected, cancelled and pending
  `would_block` evaluations plus denominator exclusion for `would_allow`.

### FocusFire peer completion uses engagement history, not death-frame orders

- **Evidence:** SC2 may clear or retarget actor orders in the same observation
  that reports the target's dead tag. The verifier previously required each
  peer to remain exact-bound in that death frame.
- **Impact:** a valid assisting attack could remain pending and later become
  `combat_target_lost`, contaminating tactical failures, repeated-error
  signatures and Playbook learning.
- **Root cause:** `current_actor_order_bound` was used for both live causal
  damage attribution and engagement membership. These are different facts.
- **Correction:** every concurrent target cohort receives one stable
  engagement ID. The verifier records the last exact-bound loop and the loop at
  which an order replacement becomes confirmed. Target death produces one
  unique kill; same-engagement peers that were ever exact-bound and were not
  confirmed replaced before death terminate neutrally as
  `satisfied_by_peer`. A current exact-bound command is preferred as the unique
  kill claimant when observable.
- **Acceptance evidence:** regressions cover same-frame order clearing,
  temporary actor disappearance and a peer that was conclusively replaced
  before target death.

## Resolved and removed from the open register

The following previously registered failures are verified as resolved and are
no longer open issues:

- feature-action camera, selection and translator ownership in the Raw path;
- duplicate command dispatch, friendly target and unattributed primitive
  failures;
- threat level remaining low during real combat;
- global same-type retreat membership and hull-only Protoss durability;
- the HIMA `Void Ray` spelling mismatch;
- production producer provenance and acceptance-only production success;
- Builder participation in Defense movement;
- Defense-owned execution feedback entering Offense state;
- Playbook evolving/frozen configuration ambiguity and per-launch reset;
- resource-tag aliases bypassing an expansion site's canonical cluster
  quarantine;
- censored or failed episodes directly promoting executable rules.

The latest run confirms that threat assessment now reaches `high` and
`critical`, production effect confirmation is 100%, a second Nexus can be
confirmed, and evolving Playbook state persists across games and applies
non-zero score changes.

## Architecture decision

LLM-PySC2 remains responsible for SC2 process startup and observation
acquisition. RTSCortex owns semantic decisions, Raw Actions and effect
verification:

```text
LLM-PySC2 startup + observation
  -> Situation / Race Brain / Role Agents
  -> Strategic Intent Arbiter
  -> Fast Executor / Validator
  -> RawPlacementService / PySC2 Raw Action
  -> EffectVerifier
```

The remaining failures are state ownership, atomic placement legality,
idempotent tactical control, strategic resource use and event-journal
throughput problems. They are not reasons to restore feature-layer execution.

## 2026-07-28 audit reconciliation

The external audit has been checked against the current `dev` implementation,
not only against the latest run artifacts. The repair change set implements the
code corrections for all nine registered issues, but the issues remain open
until their natural-terminal acceptance criteria are measured again. Unit,
integration and contract tests are not used as substitutes for live SC2
evidence. The audit also identified one shared architectural defect and ten
hidden failure paths. Those findings are folded into the issue entries below
rather than recorded as disconnected symptoms.

The shared defect is the absence of a stable semantic operation identity across
the complete path:

```text
tick-local Intent
  -> semantic actor string
  -> candidate
  -> command attempt
  -> concrete RAW actor tags
  -> concrete target / world footprint
  -> effect verdict
```

Current `intent_id` and `command_id` values include `step_id` and/or
`game_loop`, so the same continuing operation receives a new identity on a
later observation. Conversely, the current busy-actor lock uses only a semantic
actor name, while `StrategicIntent.continuity_key` omits both the concrete actor
and concrete target. Command lifecycle tracking is therefore exact for one
attempt but cannot express that multiple attempts belong to one continuing
semantic operation.

Before the issue-specific fixes, introduce:

```text
OperationKey
  run / episode
  role and semantic objective
  semantic actor
  concrete actor-tag set or producer/builder tag
  action family
  concrete target identity or target epoch

AttemptKey
  OperationKey
  attempt ordinal
  command_id
```

Specialized identities derive from `OperationKey`:

- `PlacementReservationKey`: ability, builder tag, footprint and emitted world
  target;
- `RetreatCommitmentKey`: actor-tag set, threat signature and destination;
- `EngagementKey`: actor-tag set, ability and target tag;
- `ExpansionGoalKey`: desired base count, map/candidate epoch and race.

An operation survives observation ticks; command IDs remain attempt-level
children. Operation terminal state, concrete actor lease, target lease and
effect evidence must be shared by Runtime, Bridge and Verifier. This contract is
the prerequisite for SCX-PT-030, 032, 035, 036 and 037.

## 2026-07-28 repair implementation status

The repair order has been implemented in code in the following dependency
order. “Implemented” below means the cross-layer contract and deterministic
tests exist; “live pending” means the historical failure metrics in this file
must not be cleared until a new natural-terminal run proves the acceptance
gate.

| Area | Implemented correction | Verification status |
|---|---|---|
| Cross-layer identity | Stable `OperationKey`, attempt-level `AttemptKey`, placement, retreat, engagement and expansion keys; operation/attempt IDs now cross Intent, lineage, command, route and execution report boundaries | Python characterization tests pass; live lineage conservation pending |
| SCX-PT-030 | Exact builder-tag lease, immutable dispatch reservation, builder order baseline, one rounded-and-revalidated emitted world target, reservation evidence in EffectVerifier and terminal release | RAW Python 3.9 and effect tests pass; `accepted_without_any_start_evidence == 0` pending live |
| SCX-PT-031 | One global permanent spatial exclusion ledger, separate temporary per-action suppression, failure-class-aware release/quarantine and bounded terminal cleanup | placement tests pass; long-run boundedness and finite discovery counts pending live |
| SCX-PT-035 | Actor-tag-local retreat commitment with threat signature, arrival edge, visibility grace, hysteresis, cooldown and obsolete transition | tactical tests pass; repeated-arrival count pending live |
| SCX-PT-036 | Actor/target engagement lock, unchanged-order suppression, actor order evidence and one-to-one health-delta consumption | combat tests pass; unique engagement/order-refresh metrics pending live |
| SCX-PT-034 | Background batched SQLite writer, SQLite-first JSONL mirror, callbacks outside durable locks, WAL/indexing, explicit terminal barriers, complete post-game pagination, performance counters and Runtime/Cortex snapshots every 224 loops with tail-only recovery | restart/crash tests pass; loops/s and disk-reduction gates pending live |
| SCX-PT-032 | Durable desired-base-count `ExpansionGoalState`, candidate/scouting epoch, exhausted epoch and terminal success/exhaustion/cancellation state | expansion tests pass; multi-expansion natural-terminal behavior pending live |
| SCX-PT-037 | Defense dispatch binds exact command/operation, Shield Battery is support rather than anti-air damage, and a shared hard structure budget counts completed, constructing, reserved and selected structures across roles | Defense/structure-budget tests pass; response latency and resource-use quality pending live |
| SCX-PT-038 | Cumulative HIMA actions compile to compact counted goals; generated output is capped at 512 tokens; executable prefix is at most five steps and varies with `horizon_seconds`; independent unsupported future nodes are skipped while real unsupported dependencies block | parser, macro and Ensemble tests pass; live rejection and journal-size deltas pending |
| SCX-PT-039 | Independent paired and sequential-learning modes are separate; baseline hashes and arm ordering are checked; a machine-readable analyzer enforces terminal, immutability, carry-over, false-block, repeat-error and win-rate gates | shell/analyzer checks pass; experiments have not yet been run |

Two limitations are deliberately still visible:

1. Full `ObservationEnvelope` events remain lossless because report, replay,
   corpus and policy-shadow consumers currently depend on that public journal
   contract. Persistence is off the SC2 thread and restart work is bounded, but
   the `natural-run disk usage reduced by >= 4x` gate must be measured before a
   later typed observation-delta migration is justified.
2. A placement reservation is frozen at the Bridge's final pre-dispatch
   selection, where the exact living builder tag and authoritative RAW target
   first coexist. Runtime candidate provenance is retained across routing, but
   the Python 3.11 Runtime does not construct a Python 3.9 Bridge reservation
   object speculatively.

## Current acceptance snapshot

| Gate | Current result | Status |
|---|---:|---|
| Runtime crash | 0 | pass |
| Candidate outside dispatch | 0 | pass |
| Duplicate dispatch | 0 | pass |
| Friendly target | 0 | pass |
| Unattributed primitive | 0 | pass |
| Terminal report exactly once | 100% | pass |
| Command lineage | 100% | pass |
| Role lineage | 100% | pass |
| Builder Defense movement | 0 | pass |
| Production confirmation | 79/79 = 100% | pass |
| Threat reaches high/critical | yes | pass |
| Second Nexus confirmed | yes | pass |
| Evolving Playbook hash changes | yes | pass |
| Playbook non-zero score effect | 21,531 applications | pass |
| Build confirmation | 34/44 = 77.3% | **fail** |
| Build timeout/failure rate | 10/44 = 22.7% | **fail** |
| Tactical order idempotency | repeated movement and attack | **fail** |
| Effective live speed | 0.725 game loops/s | **fail** |
| Latest frozen/evolving paired three-seed validation | not run | **not verified** |

## Open issues

### SCX-PT-030: accepted Raw build commands can still fail to start

- **Priority:** P0
- **Components:** RawActionBridge, RawPlacementService, BuildEffectVerifier,
  Builder ownership
- **Evidence:**
  - all 44 build commands reached PySC2 acceptance;
  - only 34 produced the expected new structure;
  - the remaining 10 terminated as `no_build_start_evidence`:
    - 4 Shield Batteries;
    - 2 Cybernetics Cores;
    - 2 Pylons;
    - 1 Nexus;
    - 1 Stargate;
  - minerals, gas and prerequisites were sufficient at dispatch;
  - the exact builder order, builder approach and target structure were absent
    in the failed cases;
  - resource debit alone confirmed none of the failed commands.
- **Impact:** build effect confirmation remains below the 90% engineering gate.
  Technology, production, expansion and Defense plans can all stall after an
  apparently accepted command.
- **Root cause:**
  1. `RawPlacementService.candidates()` returns transient arguments and
     provenance, but the first stored placement object is only created later in
     `resolve()` during dispatch. Candidate validation therefore does not lease
     a placement.
  2. The Runtime busy lock is keyed by semantic actor string. The Raw executor
     resolves that actor to current unit tags and blindly selects
     `actor_tags[0]`; it does not lease an exact idle Probe and its current
     order from candidate time through dispatch.
  3. The placement target is validated as a floating-point world coordinate,
     then `_raw_point()` rounds it to integers for PySC2. The Verifier
     preferentially reads the original float from
     `RawPlacementService.command_target()`, so validation, emission and
     verification do not consume the same coordinate.
  4. `RoutedCommand.to_dict()` omits `screen_world_target` and
     `screen_anchor_tag`. Any route passing through that serialization boundary
     silently loses placement identity and may reconstruct it from a later
     Observation.
  5. PySC2 action acceptance proves only that the Raw request was submitted. It
     does not prove that SC2 installed the ability order on the exact Probe, and
     another controller may replace that order before the next observation.
  6. The verifier correctly refuses to use resource debit alone, but a missing
     exact builder order currently waits for the broad timeout instead of
     terminating or retrying the attempt through a typed start-evidence stage.
- **Required correction:**
  - create an immutable `RawPlacementReservation` at candidate selection, not
    at dispatch. It must contain `OperationKey`, reservation ID, observation
    revision, exact builder tag and order snapshot, ability ID, footprint,
    power requirement, anchor/cluster ID, and expiry;
  - choose one canonical emitted world coordinate and use that exact value for
    final validation, PySC2 dispatch and effect matching; do not retain a
    separate unrounded verifier target;
  - add a builder-tag lease. Gas management, auto-worker management and other
    build operations must exclude leased tags until terminal release;
  - require candidate generation, final validation, Raw dispatch and
    EffectVerifier preparation to consume the same reservation;
  - serialize all reservation metadata explicitly; `RoutedCommand.to_dict()`
    must round-trip the world target and anchor without reconstructing them from
    a newer Observation;
  - query or conservatively verify exact building placement and builder ability
    immediately before dispatch;
  - after PySC2 acceptance, transition through
    `accepted -> builder_order_seen/build_site_occupied -> structure_seen`.
    Missing evidence must produce a typed retryable or terminal verdict;
  - retain the existing double-evidence rule: resource debit alone must never
    prove build success.
- **Acceptance criteria:**
  - validation target equals emitted target equals verifier target for 100% of
    builds;
  - accepted builds have exact builder-tag provenance and one builder lease;
  - every accepted build has one explicit start-evidence verdict;
  - unrelated resource spending confirms 0 builds;
  - build effect confirmation is at least 90%;
  - build failure/timeout rate is at most 10%;
  - an attempt that produces no start evidence is not repeated without a new
    reservation revision, builder or target.

### SCX-PT-031: placement availability, occupancy and quarantine are still split

- **Priority:** P0
- **Components:** RawPlacementService, observation action projection, Builder,
  placement diagnostics
- **Evidence:**
  - some builds were dispatched from observations that already contained
    `placement_unavailable:<action>:occupied_or_unreachable` or
    `no_unoccupied_resource_cluster`;
  - world target `[53.0625, 77.0]` was reused by a failed Shield Battery and a
    later failed Pylon;
  - late observations reported all ordinary Protoss build actions as
    `occupied_or_unreachable` even though macro and Defense agents continued to
    request construction;
  - pre-dispatch rejection remained 0, so this disagreement was not surfaced at
    the final safety boundary.
- **Impact:** the Planner, CandidateCompiler and Raw dispatcher can disagree
  about whether a target is legal. Different action names can bypass a failed
  coordinate's quarantine, and “no legal space” does not reliably trigger
  controlled reveal or resampling.
- **Root cause:**
  1. action projection, placement diagnostics and final dispatch do not read one
     authoritative placement snapshot;
  2. `_quarantined_targets` is keyed by action name, so a footprint rejected for
     one building can be offered to another building;
  3. every failed build report and every pre-dispatch build exception currently
     calls the same permanent quarantine path. A missing builder or temporary
     visibility failure can therefore permanently poison a legal coordinate;
  4. occupancy, an active reservation, a transient retry suppression and a
     permanent terrain quarantine are represented as if they were the same
     state;
  5. `_command_targets` and quarantine entries have no explicit terminal
     cleanup, TTL or episode ownership contract;
  6. visibility exhaustion is reported, but no bounded placement-discovery
     controller owns the transition from “no legal target” to “reveal, resample
     or declare finite exhaustion”.
- **Required correction:**
  - make `RawPlacementService` the only source of build availability and
    diagnostics;
  - maintain separate ledgers for observed occupancy, active reservations,
    retryable suppressions and permanent terrain exclusions;
  - classify failures before mutating placement state:
    - `builder_unavailable`, stale observation and temporary visibility release
      the reservation without quarantining the footprint;
    - occupied, unpathable or permanently unreachable terrain retires the
      relevant world cells globally across building types;
    - power/prerequisite failures remain action-specific and expire when the
      underlying revision changes;
  - invalidate all candidates generated from an older placement revision when
    occupancy or power changes;
  - release command-to-reservation state on succeeded, failed, cancelled,
    unconfirmed and episode terminal; expose bounded counts for every ledger;
  - add a bounded placement-discovery state machine that either finds a new
    legal region or emits a typed finite-exhaustion result.
- **Acceptance criteria:**
  - an action marked placement-unavailable in the current revision is dispatched
    0 times;
  - `builder_unavailable` permanently quarantines 0 legal coordinates;
  - a permanently invalid footprint is dispatched at most once across all
    building actions;
  - every unavailable build frontier has one typed reason;
  - reveal/resample has a finite attempt budget and one terminal result;
  - placement reservation and quarantine memory return to a bounded steady
    state during a long episode;
  - placement reason counts appear in report and Live Console.

### SCX-PT-032: expansion commitments re-arm after success or finite exhaustion

- **Priority:** P1
- **Components:** EconomyAgent, expansion scouting, RawPlacementService,
  GoalProgress, commitment lifecycle
- **Evidence:**
  - the first commitment successfully confirmed a second Nexus at loop 7,753;
  - a later Nexus attempt at `[17, 26]` failed with
    `no_build_start_evidence`;
  - four expansion commitments were created in the same episode:
    - one ended as `nexus_effect_confirmed`;
    - two ended as `expansion_candidates_exhausted`;
    - one remained active until episode-end cancellation;
  - one exhausted commitment was followed by another commitment that
    immediately observed the same exhausted waypoint state.
- **Impact:** the system can repeatedly spend decision bandwidth on an already
  satisfied or finitely exhausted expansion goal. Terminal commitment counts no
  longer correspond one-to-one with strategic expansion objectives.
- **Root cause:**
  1. `_ensure_expansion_commitment()` starts a commitment whenever a newly
     accepted HIMA proposal contains the townhall macro action. The trigger is a
     proposal token, not a persistent strategic base-count goal.
  2. resource clusters and failed sites are canonical, but the higher-level
     desired-base-count objective is not persistent;
  3. a terminal commitment clears the active object without retaining a durable
     `satisfied`, `exhausted` or `cooldown` result keyed by desired base count
     and candidate generation;
  4. exhausted state is transported from the Worker to Runtime through parsed
     alert strings rather than one typed expansion state contract;
  5. a later HIMA/Economy proposal can therefore instantiate the same semantic
     `BUILD NEXUS` objective without new map evidence or a higher desired base
     count.
- **Required correction:**
  - make expansion policy state explicit:
    `desired_base_count`, `confirmed_base_count`, candidate epoch, retired
    clusters, exhaustion reason and retry cooldown;
  - derive an `ExpansionGoalKey` from desired base count and candidate epoch;
    bind exactly one operation/commitment to that key;
  - replace alert-string parsing with a typed Worker/Runtime expansion scouting
    payload;
  - after success, require a higher desired count before creating another
    commitment;
  - after finite exhaustion, require new map evidence, a new cluster epoch or
    an explicit strategic override before retrying.
- **Acceptance criteria:**
  - one desired-base-count increment produces exactly one terminal commitment;
  - a confirmed or permanently failed cluster is never attempted again;
  - finite exhaustion cannot immediately create an identical commitment;
  - every expansion request ends as confirmed, finitely exhausted or explicitly
    cancelled.

### SCX-PT-034: live event durability still limits the game to 0.725 loops/s

- **Priority:** P0
- **Components:** EventStore, SQLite, JSONL journal, report generation, Live
  Console persistence
- **Evidence:**
  - 27,294 game loops required 10 h 27 min 52 s;
  - the effective rate was 0.725 game loops/s, below the 2 loops/s acceptance
    floor and far below SC2's nominal 22.4 loops/s;
  - the episode generated 250,234 events;
  - artifacts consumed approximately:
    - 1.1 GB for `events.sqlite3`;
    - 945 MB for `events.jsonl`;
    - 77 MB for `timeline.md`;
  - SQLite used `journal_mode=delete` and `synchronous=FULL`;
  - `append_event()` flushes when the elapsed interval exceeds 0.25 seconds.
    Since one live tick already exceeds that interval, the first event of almost
    every tick still performs a synchronous commit.
- **Impact:** natural-terminal regression is about 31 times slower than
  nominal game time, multi-seed validation is expensive, and a 27-game matrix
  would generate excessive wall time and disk usage.
- **Root cause:**
  1. batching was validated with a continuous-event microbenchmark, but the
     live workload is a burst followed by an interval longer than the flush
     threshold;
  2. `append_event()` performs SQLite insertion, JSONL serialization/write and
     time-based commit/flush on the SC2 control thread;
  3. `_publish()` invokes every subscriber while the main append lock remains
     held. The subscriber contract says “enqueue only”, but the store does not
     enforce or isolate that property;
  4. SQLite and JSONL are two independently written authorities. A crash between
     the two writes can leave them inconsistent, while recovery does not define
     which copy wins;
  5. SQLite `DELETE/FULL` durability on scratch turns each tick-level commit
     into a blocking filesystem transaction;
  6. every tick stores a full Observation plus high-frequency Situation, role,
     arbitration and executor projections in SQLite and JSONL, then expands
     them again into a verbose Markdown timeline;
  7. recovery calls `events_of_type()` repeatedly over the complete episode.
     The current index does not include `event_type`, so restart work grows with
     journal length;
  8. post-game Playbook review calls `events_after(..., limit=100_000)` once.
     Episodes such as the latest 250,234-event run are silently reviewed from a
     truncated prefix rather than through complete pagination.
- **Required correction:**
  - separate a bounded real-time telemetry channel from the durable semantic
    event channel. RGB/unchanged projections are lossy; command lifecycle,
    operation lifecycle, execution and terminal events are lossless;
  - assign monotonic event IDs before enqueue and move all durable writes to one
    background writer;
  - choose one canonical durable authority. JSONL should be generated/exported
    from it or written by the same commit protocol, not treated as a second
    independent source of truth;
  - move subscriber callbacks outside the store lock and give each sink a
    bounded queue with dropped/resync metrics;
  - perform interval flushing only in the writer and use an explicit
    episode-terminal barrier before review, report or shutdown;
  - store full Observation snapshots at bounded checkpoints and use typed deltas
    between them;
  - add `(run_id, episode_id, event_type, event_id)` indexing and compact
    recovery snapshots/checkpoints; recovery must not rebuild all state from
    unbounded scans;
  - paginate the complete event stream for post-game review or consume a
    terminal snapshot plus semantic events;
  - make full Markdown timeline generation optional and provide a compact
    default report;
  - profile tick latency before and after the change so event persistence is
    measured rather than assumed to be the only remaining speed limiter.
- **Acceptance criteria:**
  - `append_event()` performs no synchronous filesystem commit;
  - no subscriber executes while the durable writer lock is held;
  - close/terminal loses 0 events and event ordering remains exact;
  - injected crash tests cannot produce divergent authoritative histories;
  - reconnect sees committed events within one second;
  - post-game review consumes 100% of semantic events for episodes with more
    than 100,000 total events;
  - restart recovery work is bounded by the latest checkpoint plus a configured
    tail, not total episode length;
  - live speed is at least 2 game loops/s;
  - one equivalent natural-terminal run uses at least four times less disk;
  - deterministic Mock decisions are unchanged with persistence enabled or
    disabled.

### SCX-PT-035: actor-local retreat arrival is not an idempotent terminal state

- **Priority:** P0
- **Components:** RetreatAgent, Tactical state machine, MoveEffectVerifier,
  Intent Arbiter
- **Evidence:**
  - `CombatGroup8/Phoenix-1` produced 477 `retreat_arrived` state events between
    loops 8,160 and 18,957;
  - the same actor produced 477 successful Retreat `Move_Minimap` commands;
  - consecutive observations can emit the same arrival transition again;
  - smaller repeats also occurred for Adept and Void Ray actors.
- **Impact:** the system spends actions and journal volume repeatedly proving
  that an actor has already arrived. Retreat activity and aggregate success are
  inflated, while the actor can be prevented from returning to useful offense.
- **Root cause:**
  1. recovery is evaluated before `overwhelmed`. A shield-recovered actor can
     have its state deleted and immediately recreated by the still-active
     critical threat in the same tick;
  2. any temporary absence from current `Move_Minimap` actor scopes immediately
     deletes retreat state. Camera/selection/availability gaps therefore look
     like semantic actor disappearance;
  3. state is keyed only by semantic actor string, not the concrete RAW unit-tag
     membership, threat signature and destination;
  4. every successful Move observed while retreat state exists emits
     `retreat_arrived`; the event is not restricted to a
     `retreating -> holding` edge;
  5. a persistent retreat condition creates a new tick-local Intent and command
     identity, and neither the requested destination nor the current SC2 unit
     orders are compared with the previous operation.
- **Required correction:**
  - persist a `RetreatCommitmentKey` containing semantic actor, concrete tag
    set, threat signature and destination;
  - compute durability, current threat and `overwhelmed` first; release only
    when recovery hysteresis is satisfied and the original threat is no longer
    active;
  - tolerate bounded observation/action-scope gaps and reconcile membership by
    stable unit tags before expiring state;
  - make arrival an edge-triggered `retreating -> holding` transition, followed
    by cooldown and one explicit obsolete/re-arm reason;
  - re-arm only when threat signature, destination or concrete actor membership
    materially changes;
  - suppress a move when the actor is already at the destination or already has
    the same live order.
- **Acceptance criteria:**
  - one actor/threat/destination commitment emits at most one successful arrival;
  - identical retreat movement is not re-dispatched on consecutive ticks;
  - repeated `retreat_arrived` events are 0;
  - a one-observation actor-scope gap destroys 0 active retreat commitments;
  - shield recovery during an unchanged overwhelming threat does not
    release/recreate the operation;
  - Retreat command counts represent distinct navigation commitments.

### SCX-PT-036: focus-fire reissues identical attack orders and inflates success

- **Priority:** P1
- **Components:** FocusFireAgent, CombatEffectVerifier, actor order ledger,
  evaluation metrics
- **Evidence:**
  - Void Ray focus fire produced 1,504 `Attack_Unit` reports;
  - one target received 101 separately tracked successful commands;
  - failures still included 75 `combat_target_lost` and 14
    `combat_effect_not_observed`;
  - repeated damage ticks can confirm many commands for one continuous
    engagement.
- **Impact:** meaningful-command success overstates tactical quality, event
  volume grows unnecessarily, and the Fast Executor keeps replacing an already
  valid SC2 attack order.
- **Root cause:**
  1. `TacticalAgent._intent()` includes `step_id` in identity, so every
     observation creates a new Attack intent even when actor, target and ability
     are unchanged;
  2. the actor order ledger does not expose a stable
     `(concrete actor tags, ability, target tag)` commitment to
     CandidateCompiler;
  3. Strategic continuity is only
     `(role, action, target kind, region)`. It omits semantic actor, concrete
     actor tags and target tag, so Arbiter commitment cannot deduplicate one
     engagement;
  4. CombatEffectVerifier stores a target baseline but no actor tags or order
     identity. Any later health reduction can independently confirm every
     pending command for that target;
  5. TacticalAgent waits for `confirmation_kind == "target_removed"` to retire a
     known enemy structure, but CombatEffectVerifier never emits that value.
     Target disappearance eventually becomes `combat_target_lost`.
- **Required correction:**
  - keep one `EngagementKey` keyed by concrete actor-tag set, ability and target
    tag; tick-local commands are refresh attempts under that engagement;
  - emit a new Raw attack only when the target changes, the order disappears,
    the command times out or a higher-priority Intent preempts it;
  - record the dispatched actor tags and verify their exact Raw order where
    observable;
  - make damage evidence one-to-one with an active engagement. One health delta
    cannot terminalize multiple independent operations;
  - define target disappearance semantics explicitly:
    confirmed death/removal, temporary visibility loss and stale last-known
    target must remain distinct;
  - model CombatEffectVerifier around one engagement lifecycle rather than one
    report per observation;
  - report both unique engagements and low-level order refreshes.
- **Acceptance criteria:**
  - unchanged actor/target attack orders are not reissued on consecutive ticks;
  - one engagement has one terminal effectiveness result;
  - one target health delta confirms at most one engagement operation;
  - Tactical and Verifier use the same target removal/loss enum;
  - command success metrics exclude transport-level order refreshes;
  - target-lost and no-effect failures remain fully classified.

### SCX-PT-037: Defense is operational but strategically overproduces static defense

- **Priority:** P1
- **Components:** DefenseAgent, ResourceClaim, Strategic Intent Arbiter,
  RaceProfile defense policy
- **Evidence:**
  - 15 Shield Battery builds were attempted; 11 succeeded and 4 failed;
  - the final state contained 11 Shield Batteries and 12 Pylons, but only one
    Stargate and four Warp Gates;
  - the game ended with 7,820 minerals and 5,417 gas unspent;
  - post-game review still identified one 112-loop `critical` threat interval
    with no effective executed response;
  - the final army was air-heavy: 9 Void Rays and 15 Phoenixes.
- **Impact:** Defense now emits real production, static-defense and combat
  Intents, but it can consume placement attempts without creating a balanced
  response. Large resource banks and insufficient production scaling reduce
  strategic quality even when VeryEasy is defeated.
- **Root cause:**
  1. emergency production/static-defense branches append intents directly and
     bypass the `_may_emit()`/`_activate_actor()` operation state used by the
     routed tactical branch;
  2. the structure saturation check in `CortexRuntimeEngine` applies only to
     HIMA macro frontiers. Defense-owned build intents do not pass through that
     guard;
  3. Protoss RaceProfile classifies Shield Battery as both ordinary static
     defense and anti-air defense, even though it does not itself damage air
     units. Air response selection can therefore prefer repeated Batteries over
     mobile anti-air production;
  4. emergency compilation reacts to current threat but lacks a persistent
     desired-defense inventory per base and threat signature;
  5. static defense does not claim opportunity cost against production capacity,
     army composition and expansion;
  6. `DefenseAgent.record_execution()` searches current state by actor/action
     rather than consuming an exact operation/command lineage. Repeated same
     action objectives can receive ambiguous feedback;
  7. the Arbiter scores urgency but does not penalize excessive static-defense
     saturation or prolonged unspent resources.
- **Required correction:**
  - require every Defense proposal branch to use the same operation ledger and
    exact execution lineage;
  - define RaceProfile defense templates with per-base/per-threat target counts;
  - distinguish healing/support structures from anti-air damage sources in the
    doctrine; an air threat must request a mobile or damaging anti-air response
    before optional support;
  - enforce structure saturation as a shared Candidate/Validator invariant,
    counting complete structures, construction, active reservations and
    selected intents from all roles;
  - deduplicate static-defense objectives by operation key, base location, type
    and threat signature;
  - add opportunity-cost and saturation terms to ResourceClaim/Arbiter scoring;
  - let Defense request production scaling or mobile counters when static
    defense is already sufficient;
  - require every critical-threat response to record whether it reduced the
    threat, merely delayed it or failed.
- **Acceptance criteria:**
  - static-defense count follows an explicit RaceProfile target rather than
    repeated tick-level requests;
  - all Defense branches have one operation state and exact command feedback;
  - completed + constructing + reserved static defenses never exceed the
    applicable hard cap;
  - Shield Battery is never counted as the sole anti-air damage response;
  - one defense objective has one commitment and terminal result;
  - critical threats receive a selected response within 16 loops;
  - prolonged large resource banks trigger a supported production, expansion or
    explicit `no_runtime_spend_available` diagnosis;
  - Defense quality is reported separately from raw command success.

### SCX-PT-038: HIMA cumulative plans remain too long and contain unsupported frontiers

- **Priority:** P1
- **Components:** HIMA parser, MacroPlan compiler, RaceProfile capability
  mapping, report
- **Evidence:**
  - the Ensemble completed 567 macro requests with no degraded members;
  - 13 macro proposals were rejected as
    `unsupported_by_runtime:not_implemented`;
  - accepted parser output can still contain 34-58 expanded logical steps;
  - examples include currently unsupported Fleet Beacon, Robotics Facility,
    Robotics Bay, Photon Cannon, Carrier and Observer actions;
  - long cumulative outputs are persisted repeatedly in the event journal.
- **Impact:** useful supported actions later in a proposal can be hidden behind
  unsupported future actions. Planning records are unnecessarily large, and
  rejection can reflect Runtime coverage rather than poor HIMA strategy.
- **Root cause:**
  1. parser constants (`128` logical items, repeat `32`, expanded count `256`)
     are safety ceilings, not a live planning horizon;
  2. the local HIMA class defaults to 2,048 generated tokens, while current live
     Protoss configs explicitly use 512. The audit concern is therefore not the
     active token setting alone: even a valid 512-token cumulative plan can be
     much longer than the immediately executable horizon;
  3. `MacroPolicyProposal.horizon_seconds` is stored but never consulted by the
     live mapper/compiler;
  4. HIMA describes cumulative desired inventories, while the compiler projects
     every parsed step into a sequential `MacroPlan`; counted targets are only
     locally converted to repeats;
  5. `runtime_frontier()` deliberately treats an early non-managed unsupported
     action as a hard blocker, so later supported work can be hidden even when
     it is not dependency-related;
  6. Runtime-owned Protoss capability covers only a subset of the official HIMA
     vocabulary, but the proposal has no explicit dependency DAG to distinguish
     “unsupported prerequisite” from “independent unsupported future advice”.
- **Required correction:**
  - retain cumulative targets as compact counted objectives;
  - compile a typed macro dependency DAG using RaceProfile prerequisites and
    mutually exclusive transitions;
  - project only a configurable bounded executable frontier from that DAG;
    `horizon_seconds` and plan TTL must have defined, tested semantics;
  - preserve later supported independent actions when an earlier future action
    is unsupported, but block any node whose real prerequisite is unsupported;
  - distinguish unsupported capability from illegal or failed strategy in
    reports;
  - expand Protoss Runtime coverage only through RaceProfile specs and complete
    effect verification.
- **Acceptance criteria:**
  - live projected MacroPlan length and repeat expansion are bounded and
    deterministic by configuration;
  - full HIMA output remains available for provenance without expanding the hot
    event path;
  - an unsupported future action does not reject an otherwise useful proposal;
  - an unsupported true prerequisite cannot be silently skipped;
  - changing `horizon_seconds` changes the projected horizon in a tested way;
  - unsupported, deferred, future and illegal classifications remain mutually
    exclusive and conserved.

### SCX-PT-039: the latest architecture lacks paired multi-seed acceptance evidence

- **Priority:** P1
- **Components:** regression harness, evolving/frozen Playbook experiment,
  engineering acceptance report
- **Evidence:**
  - the latest `ce9d095` validation contains one completed seed: evolving
    Playbook seed 2;
  - evolving mode is proven to mutate and persist its database;
  - current Playbook applications include 21,531 non-zero score deltas and 13
    blocks;
  - no matched seed `[0,1,2]` frozen/evolving run set exists after the latest
    Raw placement, Defense and event-store changes;
  - false-block rate and repeat-error reduction therefore cannot be measured
    causally from the current result.
- **Impact:** one VeryEasy victory proves feasibility, not stability,
  repeatability or Playbook benefit. It cannot close the Protoss acceptance
  phase or justify starting the final 27-game matrix.
- **Root cause:** long wall time and repeated P0 runtime defects have forced
  validation to proceed one seed at a time. In addition, the current
  `run_protoss_playbook_paired_natural_terminal.sh` mixes two different
  questions:
  - frozen starts from a fresh baseline for every seed;
  - evolving starts once from baseline and carries seed 0 state into seed 1 and
    seed 2;
  - arm order is always frozen then evolving;
  - the harness records exit code and database hashes, but does not calculate
    false blocks, eligible repeated errors, strategic consequences or matched
    outcome deltas.
  The current script is useful as a sequential-learning smoke, but it is not an
  independent matched-pair causal experiment.
- **Required correction:**
  - first close SCX-PT-030, 031, 034 and 035;
  - freeze one read-only Playbook snapshot;
  - implement two explicitly separate experiment modes:
    1. **independent paired causal evaluation:** for every seed, both arms start
       from the same baseline snapshot; arm order is balanced; no learning
       crosses seed boundaries;
    2. **sequential learning evaluation:** evolving deliberately carries state
       from seed N to N+1, while frozen reference runs remain unchanged;
  - produce a machine-readable comparison report for engineering gates,
    eligible repeated-error recurrence, false blocks, rule applications,
    strategic consequences, score and outcome;
  - only then promote the Protoss configuration to the 27-game matrix.
- **Acceptance criteria:**
  - all independent paired and sequential-learning games reach a natural
    terminal without Runtime failure;
  - both independent arms begin each seed with identical baseline content;
  - frozen database hash is unchanged for every game;
  - only the sequential evolving experiment permits seed N+1 to retrieve
    evidence from seed N;
  - hard-rule false-block rate is at most 1%;
  - repeated eligible errors decrease by at least 50% without reducing matched
    seed win rate;
  - report classification and comparison counts are conserved; exit code 0 and
    a changed hash alone never constitute acceptance;
  - all P0 engineering gates pass across the aggregate.

#### 2026-07-28 acceptance-framework correction

- **Additional evidence:** the first paired harness revision used `rule_mode:
  active` for every Frozen/Evolving run, while its hard false-block gate
  required at least one resolved counterfactual. An active hard block prevents
  dispatch and therefore cannot acquire a terminal counterfactual result; a run
  with no active hard block likewise has a zero denominator. The 12-run matrix
  consequently had no production path to a passing false-block gate.
- **Additional root cause:** two different evidence domains were collapsed:
  active behavior measures the real effect of Playbook decisions, whereas a
  blocking counterfactual is observable only when the same candidate is allowed
  to pass all non-Playbook validation, wins final arbitration, is dispatched,
  and reaches one terminal outcome. Pre-arbitration candidates that were never
  selected are not missing counterfactuals. In addition, the comparison
  analyzer reported several engineering metrics but did not include the full
  SCX-PT-039 engineering contract in `accepted`, allowing a causal pass to hide
  build, tactical, recovery, speed or storage failures.
- **Implemented correction:**
  - the 12 active behavior runs remain unchanged, but every behavior arm now
    has a separate shadow calibration twin restored from that arm's exact
    pre-run Playbook snapshot;
  - `PlaybookRuleEvaluation` records a stable cross-run
    `counterfactual_key`, rule kind and `counterfactual_observable`; only a
    final dispatch marks a shadow decision observable;
  - active hard blocks must have matched resolved shadow evidence, and
    unselected pre-arbitration candidates are excluded rather than counted as
    unresolved;
  - execution-guard false blocks and strategic regret are separate metrics.
    Strategic rules use post-game strategic consequences and never treat a raw
    command success as proof that the block was wrong;
  - every run emits `engineering-gates.json`. Missing required values fail
    closed, and the aggregate gates every behavior run for lineage, production,
    build start/effect/provenance, placement identity, quarantine/redispatch,
    Retreat/FocusFire/Expansion/Defense invariants, subscriber isolation,
    bounded recovery, post-game coverage, live speed and disk reduction;
  - the formal runner now requires an expected Git SHA and a clean
    superproject worktree (the intentionally patched read-only submodule is
    recorded separately), and propagates analyzer rejection to its exit code.
- **Remaining evidence requirement:** the framework correction is covered by
  deterministic tests, but SCX-PT-039 remains open until the expanded active +
  matched-shadow run set completes and every generated causal and engineering
  gate passes. Code completion must not be reported as multi-seed empirical
  acceptance.

#### 2026-07-29 fail-closed acceptance review

- **Status:** the eight review findings A-01 through A-08 are corrected in code
  and deterministic regression tests. SCX-PT-039 itself remains open until the
  12 active behavior runs and their 12 matched shadow calibration runs complete.
- **Evidence and root causes:**
  1. the comparison aggregated `REQUIRED_ENGINEERING_GATES` only across active
     behavior rows even though shadow rows supply the causal evidence;
  2. unchanged-attack detection counted terminal reports by engagement ID, so
     one legitimate multi-actor FocusFire engagement and its
     `satisfied_by_peer` terminal could be reported as redispatch;
  3. zero production/build exposure and absent recovery events were represented
     as success rather than unobserved evidence;
  4. placement acceptance inferred identity from building action and center
     position. It did not consume the authoritative discrete footprint ledger,
     could not see pre-acceptance failures or cross-type overlap, and could not
     prove that non-spatial failures avoided permanent suppression;
  5. Defense acceptance reconstructed only completed Shield Batteries from
     observations and omitted construction, queue, reservation, dispatch and
     RaceProfile unit caps;
  6. Active/Shadow counterfactual keys represented broad situation context,
     not the same intent/candidate, actor, target, decision epoch and pre-action
     state. Missing observability and rule-kind fields were accepted through
     permissive defaults;
  7. the analyzer read each entire natural-terminal JSONL into memory and then
     traversed it repeatedly;
  8. source identity was checked only once before the long experiment and
     ignored the patched submodule. A mid-run source mutation could therefore
     retain the initial clean metadata.
- **Implemented correction:**
  - all required engineering gates and all missing values now aggregate across
    every behavior and calibration row;
  - attack redispatch identity is derived from dispatched
    `(actor, target, operation_id)` commands and remains locked only until that
    exact command reaches a terminal report. Terminal report multiplicity and
    `satisfied_by_peer` are not dispatch evidence, while a post-terminal retry
    is a new attempt;
  - production/build minimum exposure and recovery evidence are explicit
    gates. Zero denominators remain `null` and fail closed. Recovery is supplied
    by a deterministic canary artifact bound to the expected Git SHA;
  - `RawPlacementService` emits reservation, occupancy, suppression and release
    transitions with exact footprint cells. Terminal execution persists these
    transitions before publishing the report. The analyzer validates declared
    state history, cross-type cell overlap, permanent-invalid redispatch and
    spatial versus actor/non-spatial failure classes;
  - `DefenseAgent` emits authoritative `defense_inventory_evaluated` events for
    structures and units using completed, constructing/training, reserved and
    dispatched-not-terminal counts against RaceProfile hard caps;
  - candidate counterfactual identity includes canonical action arguments,
    actor, rule, decision epoch and a run-neutral pre-action observation hash.
    Intent identity includes action family, actor scopes, desired effect,
    producer/resource claims, operation and continuity. The analyzer rejects
    missing identity, observability or rule-kind fields;
  - engineering analysis uses a single-pass accumulator and retains only
    command-scale acceptance evidence rather than Observation-frequency events;
  - the formal runner records Git HEAD, superproject dirty state, submodule
    commit, submodule dirty state and binary-diff SHA before and after every
    behavior and shadow run, checks them against one baseline, checks again
    before analysis, and requires one source-attestation fingerprint across the
    full matrix.
- **Acceptance criteria added by this review:**
  - one failed or missing Shadow engineering gate rejects the full comparison;
  - zero production/build observations and missing recovery proof never appear
    as successful coverage;
  - two actors attacking one target once do not count as redispatch, while the
    same actor/target/operation dispatched twice does;
  - every accepted placement run contains legal authoritative ledger
    transitions, no overlapping active footprints and no reservation after a
    permanent-invalid cell;
  - Defense effective inventory never exceeds its emitted hard cap;
  - only an exact observable Active/Shadow candidate or intent match may resolve
    a blocking counterfactual;
  - analyzer retained-event growth is command-scale rather than
    Observation-scale;
  - any pre/post or cross-run source-attestation change rejects acceptance.

## Repair order

1. Freeze characterization tests and add the cross-layer `OperationKey`,
   `AttemptKey` and typed specialized operation keys. Do not change live
   selection until replay proves identity conservation.
2. SCX-PT-030 and SCX-PT-031: introduce exact builder/placement leases, one
   emitted coordinate, failure-class-aware ledgers and terminal cleanup.
3. SCX-PT-035 and SCX-PT-036: rebuild Retreat and Combat around actor-tag-local
   operation state, edge-triggered transitions and one-to-one effect evidence.
4. In parallel with steps 2-3, SCX-PT-034: move durable persistence off the SC2
   path, establish one authority, add terminal barriers, snapshots, complete
   pagination and performance profiling.
5. SCX-PT-032: bind expansion to a durable desired-base-count goal and typed
   scouting epoch.
6. SCX-PT-037: route every Defense branch through the operation ledger, enforce
   global saturation and correct anti-air semantics.
7. SCX-PT-038: compile compact HIMA goals into a bounded dependency-aware
   frontier.
8. Run all Python 3.11, Bridge Python 3.9 and front-end checks, including
   restart/crash/recovery tests.
9. SCX-PT-039: run independent paired evaluation and sequential-learning
   evaluation as separate acceptance sets.

## Required engineering gates

```text
runtime crash = 0
candidate outside dispatch = 0
duplicate dispatch = 0
friendly target = 0
terminal report exactly once = 100%
command and role lineage = 100%
minimum accepted production exposure >= 1 per run
production effect confirmation = 100%
minimum accepted build exposure >= 1 per run
build-start evidence coverage = 100%
build effect confirmation >= 90%
build failure/timeout <= 10%
validated placement target equals emitted and verified target = 100%
accepted build builder-tag provenance = 100%
authoritative placement ledger evidence is present and transitions are legal
cross-type active footprint overlap = 0
non-spatial failure permanent footprint quarantine = 0
invalid world footprint redispatch = 0
repeated retreat arrival = 0
unchanged actor-target attack redispatch = 0
one health delta confirms at most one engagement
expansion commitment immediate re-arm after terminal = 0
Defense completed + constructing + reserved inventory <= hard cap
authoritative Defense inventory evidence is present
subscriber callback under durable append lock = 0
post-game semantic event coverage = 100%
source-bound restart recovery evidence is present and bounded by checkpoint tail
effective live speed >= 2 game loops/s
natural-run disk usage reduced by >= 4x
frozen Playbook hash remains unchanged
evolving Playbook survives and affects the next game
independent paired and sequential-learning experiments reported separately
all behavior and shadow rows share one pre/post source attestation
```
