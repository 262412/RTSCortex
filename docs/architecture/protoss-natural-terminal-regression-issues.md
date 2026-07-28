# Protoss natural-terminal regression issue register

Status date: 2026-07-28

This register contains only issues that remain open after:

```text
protoss-raw-architecture-natural-terminal-20260726T212037Z
```

The run used HIMA Protoss a/b/c, active Strategic Intent Arbiter, the active
Playbook guard, `Simple64`, Protoss versus VeryEasy Zerg, PySC2 Raw Actions,
`game_steps_per_episode: 0`, `step_mul: 1` and
`simulation_speed_multiplier: 0.5`.

| Seed | Terminal | Game loops | Wall time | Meaningful success | Build | Production |
|---|---|---:|---:|---:|---:|---:|
| 0 | defeat | 21,103 | 7 h 11 min | 252/347 = 72.6% | 54/81 = 66.7% | 63/66 = 95.5% |
| 1 | defeat | 30,355 | 11 h 54 min | 1,630/1,826 = 89.3% | 41/74 = 55.4% | 71/72 = 98.6% |
| 2 | Slurm timeout | 11,800 | about 4 h 49 min | partial only | no terminal report | no terminal report |

The seed 1 aggregate success rate is misleading: 1,119 successful deterministic
Defense `Move_Minimap` commands dominate it. Excluding those repeated moves, the
remaining completed actions are approximately `511/688 = 74.3%`.

The following issues were verified as resolved and have therefore been removed
from the open register:

- feature-action camera/selection/translator ownership in the Raw execution
  path;
- duplicate dispatch, friendly target and unattributed primitive failures;
- threat level remaining low during real combat;
- global same-type retreat membership and hull-only Protoss durability;
- the HIMA `Void Ray` spelling mismatch;
- production producer provenance and acceptance-only production success;
- censored/error episodes directly promoting executable rules.

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

The remaining issues are not reasons to restore feature-layer execution. They
are state ownership, evidence semantics, placement discovery, persistent
learning and event-journal throughput problems above or below the Raw action
boundary.

## Open issues

### SCX-PT-029: Defense terminal feedback re-arms the same movement every tick

- **Priority:** P0
- **Components:** DefenseAgent, Tactical Agent, execution lineage
- **Evidence:**
  - seed 1 emitted 1,233 deterministic Defense commands;
  - 1,138 were `Move_Minimap`, including 1,027 for
    `CombatGroup0/Zealot-1` and 62 for `Builder/Builder-Probe-1`;
  - 1,119 Defense moves succeeded;
  - the Tactical Agent recorded `offense_arrived` for Defense-owned movement,
    so one Agent's terminal feedback mutated another Agent's state.
- **Impact:** event volume, command success and tactical activity are dominated
  by artificial movement. The Builder is pulled away from construction and
  aggregate action success no longer measures useful play.
- **Root cause:**
  1. `DefenseAgent.record_execution()` deletes a successful state immediately,
     so a persistent threat recreates the identical rally intent on the next
     observation;
  2. the Defense movement loop accepts every actor exposed by
     `Move_Minimap`, including Builder;
  3. `CortexRuntimeEngine.record_execution()` sends every Attack/Move report to
     `DeterministicTacticalAgent`, even when command lineage says Defense owns
     it.
- **Required correction:**
  - restrict defensive rally movement to combat actors;
  - keep actor-local `moving -> holding -> obsolete/cooldown` state after
    successful arrival;
  - release or retarget only when the threat signature materially changes;
  - route execution feedback only to the role that owns the command.
- **Acceptance criteria:**
  - Builder Defense movement is 0;
  - identical Defense move is emitted at most once per actor/threat
    commitment;
  - Defense reports create 0 `offense_arrived` transitions;
  - Defense movement no longer contributes more than 25% of meaningful
    commands.

### SCX-PT-030: build acceptance lacks a distinct start-confirmation stage

- **Priority:** P0
- **Components:** RawActionBridge, Build EffectVerifier, report/Console
- **Evidence:**
  - seed 0 confirmed 54/81 builds; seed 1 confirmed 41/74;
  - seed 0 build failures included 15 `builder_not_observable`,
    4 `no_build_order_observed`, 5 `target_not_created` and
    3 `worker_order_replaced`;
  - seed 1 included 25 `no_build_order_observed`, mostly Shield Batteries;
  - current verifier records worker orders and mineral delta, but it only has
    accepted-or-final-effect semantics.
- **Impact:** a command that entered SC2's build workflow can time out before a
  structure is observed, while a command that never started is reported with a
  similar terminal. This prevents reliable root-cause separation and causes
  premature retries.
- **Root cause:** the state machine jumps from `PYSC2_ACCEPTED` directly to
  structure-effect confirmation. Worker order, cost debit, builder movement and
  target occupancy are diagnostic fields rather than a typed intermediate
  verdict with its own timeout policy.
- **Required correction:**
  - implement
    `PYSC2_ACCEPTED -> START_EVIDENCE_PENDING -> BUILD_STARTED_CONFIRMED ->
    EFFECT_CONFIRMED`;
  - treat exact builder order or a new expected structure tag as authoritative
    start evidence;
  - allow the supporting quorum `resource debit + builder approach/target
    occupancy`, but never use resource debit alone because concurrent spending
    is common;
  - extend the effect deadline only after start confirmation;
  - if start was confirmed but no target structure appears, return
    `build_started_effect_missing`; otherwise return
    `no_build_start_evidence`;
  - expose all evidence and confirmation loops in report and Console.
- **Acceptance criteria:**
  - every accepted build has an explicit start-confirmation result;
  - unrelated resource spending alone confirms 0 builds;
  - build-start evidence coverage is 100%;
  - `no_build_order_observed` is replaced by the two causal terminal classes;
  - build effect confirmation is at least 90% and timeout at most 10%.

### SCX-PT-031: no legal visible placement is hidden instead of diagnosed

- **Priority:** P0
- **Components:** RawPlacementService, observation action projection, Builder
- **Evidence:**
  - long runs repeatedly defer buildings when the current feature viewport has
    no powered, pathable and empty footprint;
  - pre-dispatch placement rejection is 0, but build proposals disappear or
    later retry similar locations without a recorded placement-exhaustion
    reason;
  - Raw execution no longer needs feature-layer camera selection, yet ordinary
    screen-build candidates are still derived from the current screen masks.
- **Impact:** the macro frontier appears resource- or model-blocked when the
  real constraint is spatial visibility. The Agent cannot deliberately reveal
  new buildable space and reports cannot distinguish `need_power`, `occupied`,
  `not_pathable` and `out_of_view`.
- **Root cause:** `RawPlacementService` owns target identity and final
  validation but does not own a map/world placement grid or a structured
  candidate-generation diagnostic. Candidate absence is represented by an
  omitted action rather than a typed state that can trigger scouting or Pylon
  placement.
- **Required correction:**
  - add placement diagnostics by action and reason;
  - use a persistent world-space placement grid when PySC2 game info exposes
    it, with current observation occupancy applied on top;
  - otherwise emit a bounded Builder-vision expansion intent instead of
    silently retrying the same viewport;
  - permanently quarantine pre-dispatch-invalid world footprints.
- **Acceptance criteria:**
  - every unavailable build frontier has one typed reason;
  - the same invalid footprint is never dispatched twice;
  - lack of visible space causes a bounded reveal/resample action;
  - placement-exhaustion and power/pathability/occupancy counts appear in
    report and Console.

### SCX-PT-032: expansion identity is a resource tag, not a resource cluster

- **Priority:** P0
- **Components:** RawPlacementService, expansion scouting, commitment lifecycle
- **Evidence:**
  - seed 1 failed a Nexus from anchor `0x100140001` at `[57,31]`;
  - it then succeeded from anchor `0x100380001` at the same `[57,31]`;
  - the successful anchor was later reused for a different target and failed;
  - multiple mineral/geyser tags can represent one expansion location.
- **Impact:** alternate member tags bypass suppression, occupied expansions can
  be attempted again, and the system cannot prove whether all distinct
  expansion sites were exhausted.
- **Root cause:** commitment and quarantine are keyed by individual resource
  tag. There is no canonical resource-cluster identity shared by discovery,
  placement, execution and EffectVerifier, and a confirmed Nexus does not
  retire the cluster.
- **Required correction:**
  - cluster neutral resources deterministically in world space;
  - assign one stable cluster ID and one canonical Nexus target per cluster;
  - bind commitment, command, quarantine and effect to that cluster ID;
  - retire a cluster after successful Nexus confirmation or permanent
    invalidation;
  - continue to the next cluster until success or finite exhaustion.
- **Acceptance criteria:**
  - one world expansion site is attempted at most once after failure or
    success;
  - alternate member tags cannot bypass quarantine;
  - one commitment terminal is either Nexus-confirmed or all cluster IDs
    exhausted;
  - at least one seed builds a second Nexus or records explicit finite
    exhaustion.

### SCX-PT-033: the launch harness resets Playbook state between games

- **Priority:** P0
- **Components:** CortexPlaybook configuration, run harness, post-game review
- **Evidence:**
  - `run_protoss_frozen_natural_terminal.sh` restores the same baseline before
    every seed;
  - within a game, seed 0 and seed 1 still emitted 19 and 21 rule updates, so
    current “frozen” means baseline-reset rather than read-only;
  - the next game therefore cannot use the previous game's new cases, lessons
    and promoted rules.
- **Impact:** the Playbook cannot serve as the user's persistent tactical
  notebook. A/B frozen results are also semantically ambiguous because the
  database mutates inside each supposedly frozen episode.
- **Root cause:** `rule_mode` controls guard application only. It does not define
  whether post-game learning may write. Persistence/reset behavior is left to
  shell scripts.
- **Required correction:**
  - add explicit `learning_mode: evolving | frozen`;
  - in `evolving`, reuse one canonical database across games and commit valid
    terminal reviews transactionally;
  - in `frozen`, disable case/lesson/rule mutation and use a read-only snapshot;
  - censored/error episodes remain queryable but promotion-ineligible;
  - record database hash before/after every episode.
- **Acceptance criteria:**
  - seed N+1 retrieves evidence written by seed N in evolving mode;
  - frozen mode database hash is unchanged;
  - evolving mode never resets between normal launches;
  - rule support is deduplicated by signature, episode and seed;
  - every applied rule remains traceable to prior run evidence.

### SCX-PT-034: per-event durable writes make game speed far below target

- **Priority:** P0
- **Components:** EventStore, JSONL journal, Console persistence
- **Evidence:**
  - seed 0 advanced at about 0.81 game loops/s and seed 1 at about 0.71;
  - seed 2 slowed to roughly 0.56-0.68 game loops/s and hit the 24-hour Slurm
    limit before natural terminal;
  - seed 2 wrote about 174,000 events for about 11,800 observations;
  - `EventStore.append_event()` commits SQLite and opens/closes the Lustre JSONL
    file once per event.
- **Impact:** `simulation_speed_multiplier` cannot achieve the configured
  effective rate, a three-seed natural-terminal run exceeds the allocation,
  and event I/O—not SC2 or GPU inference—becomes the limiting resource.
- **Root cause:** each observation produces many Situation, role, arbitration,
  candidate and executor events, and each event performs two synchronous
  filesystem durability operations. There is no bounded batch/flush boundary.
- **Required correction:**
  - keep the JSONL handle open for the EventStore lifetime;
  - batch SQLite commit and JSONL flush by a bounded event/time threshold;
  - explicitly flush at episode terminal and close;
  - preserve event ID order and immediate in-process Console publication;
  - ensure reconnect sees committed events within one second;
  - after the I/O fix, set the seed-2 validation configuration to the intended
    speed rather than compensating with a lower multiplier.
- **Acceptance criteria:**
  - EventStore does not open the journal per event;
  - committed-event visibility p95 is at most one second;
  - close/terminal loses 0 events;
  - identical Mock decisions are unchanged;
  - live seed 2 advances at least 2 game loops/s or the remaining bottleneck is
    separately measured and documented;
  - natural terminal completes within the Slurm wall-time allocation.

## Repair order

1. isolate role-owned execution state and stop Defense move flooding;
2. add build-start confirmation and placement diagnostics;
3. make expansion cluster identity canonical;
4. separate evolving and frozen Playbook learning semantics;
5. batch EventStore durability and restore effective game speed;
6. run all Python 3.11 and Bridge Python 3.9 checks;
7. rerun Protoss seed 2 to natural terminal with active Arbiter and evolving
   Playbook.

## Required engineering gates

```text
runtime crash = 0
candidate outside dispatch = 0
duplicate dispatch = 0
friendly target = 0
terminal report exactly once = 100%
role lineage = 100%
Builder Defense Move_Minimap = 0
Defense feedback entering Offense state = 0
build-start evidence coverage = 100%
build effect confirmation >= 90%
build timeout <= 10%
expansion cluster reuse after terminal = 0
evolving Playbook survives the next game
frozen Playbook hash remains unchanged
effective live speed >= 2 game loops/s
```
