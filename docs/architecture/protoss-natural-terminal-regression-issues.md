# Protoss natural-terminal regression issue register

Status date: 2026-07-26

This register contains only issues that remain open after the latest raw-action
natural-terminal regression:

```text
protoss-raw-frozen-natural-terminal-20260725T231010Z
```

It used HIMA Protoss a/b/c, active Strategic Intent Arbiter, a frozen active
Playbook, `Simple64`, Protoss versus VeryEasy Zerg,
`execution_action_space: raw`, and `game_steps_per_episode: 0`.

| Seed | Outcome | Steps | Meaningful success | Build | Production |
|---|---:|---:|---:|---:|---:|
| 0 | draw | 69,378 | 88.3% | 34/56 | 56/56 |
| 1 | defeat | 18,653 | 60.0% | 16/32 | 29/29 |
| 2 | defeat | 19,186 | 87.2% | 20/31 | 28/28 |

Production provenance, terminal-report exactly-once, duplicate-dispatch
protection, candidate-domain validation, friendly-target safety, raw movement
settlement and threat classification remained healthy. The remaining failures
were above the raw transport boundary:

- Defense emitted no production, anti-air or static-defense response before
  Nexus loss;
- expansion and ordinary build placement still had separate candidate,
  dispatch and effect target interpretations;
- a damaged member from another same-type control group could trigger or retain
  retreat state;
- HIMA emitted `Void Ray`, while the pinned vocabulary accepted only
  `VoidRay`.

The 2026-07-26 architecture repair adds a RaceProfile-driven emergency compiler,
single-owner tactical role lineage, `RawPlacementService`, actor membership plus
shield durability, and the explicit `Void Ray` alias. These changes are
deterministically tested but are not called live-accepted until another
natural-terminal seeds `[0,1,2]` run passes the criteria below.

The architecture migration was then exercised in two bounded live canaries. The
second run includes world quarantine, raw expansion generations and deterministic
idle-Probe mineral assignment:

```text
cortex-20260725T223904697203Z-06b36ed2
cortex-20260725T225215667936Z-477f2638
```

| Result | Value |
|---|---:|
| SC2 action space | RAW with feature/RGB observations |
| Meaningful terminal reports | 14 |
| Succeeded / failed / unconfirmed | 14 / 0 / 0 |
| Build effect confirmed | 6/6 |
| Production effect confirmed | 8/8 |
| Producer provenance | 100% |
| Generic translation failures | 0 |
| Camera/selection orchestration terminals | 0 |
| Unattributed primitives | 0 |
| Candidate-outside-dispatch | 0 |
| Observation watchdog recovery | 0 |

Both 2,000-loop runs produced the same 14/14 terminal classification with all
hard engineering gates passing. The second run opened its expansion commitment
with `generation_id=1` and an intentionally truncated episode emitted zero
promoted lessons or executable rule updates.

These results validate the new execution boundary, but they are not a
natural-terminal combat or expansion acceptance run. Issues below remain in the
register until their own long-run acceptance criteria are observed.

## Architecture decision: RTSCortex owns execution through PySC2 Raw Actions

The repeated feature-action failures had one common architectural cause. The
former live path contained two runtimes:

```text
RTSCortex Runtime
  -> LLM-PySC2 MainAgent
  -> camera
  -> feature-layer selection
  -> translator chain
  -> feature action
  -> SC2
```

RTSCortex owns the semantic command and its lifecycle, but LLM-PySC2 still owns
the final actor selection, camera state, primitive queue and translator abort.
Consequently the command validated by RTSCortex is not necessarily the command
that reaches SC2. Fixes to a camera retry, selected unit, control-group cache or
translator queue only move the failure to another boundary.

PySC2 supports `ActionSpace.RAW` while retaining feature-unit and RGB
observations. RTSCortex will therefore keep LLM-PySC2 for SC2 process startup,
map/bootstrap configuration and observation acquisition, but the Fast Executor
will emit PySC2 Raw Actions directly.

The new critical path is:

```text
LLM-PySC2 SC2 startup + PySC2 observation
  -> RTSCortex Runtime
  -> Strategic Intent Arbiter
  -> Fast Executor
  -> candidate and semantic validation
  -> RTSCortex RawActionBridge
  -> PySC2 Raw Action
  -> SC2
  -> EffectVerifier
```

LLM-PySC2 camera, selection, control-group and translator state are no longer
allowed to claim, mutate, delay or abort an RTSCortex command in raw mode.
Feature-action mode remains temporarily available for compatibility until
Terran and Zerg have separate raw-action acceptance runs.

## Open issues

### SCX-PT-027: build placement had multiple semantic owners

- **Priority:** P0
- **Status:** architecture implementation complete; natural-terminal acceptance pending
- **Components:** build candidate generation, RawActionBridge, BuildVerifier
- **Evidence:**
  - seed 1 confirmed only 7 of 37 build commands;
  - repeated failures targeted the same raw/world region around
    `[17.9375, 28.625]` through different screen coordinates;
  - failures included both `no_build_order_observed` and
    `target_not_created`.
- **Impact:** a bad location can be retried indefinitely under different camera
  projections, consuming the Builder, resources and strategic frontier.
- **Root cause:** candidate generation lived in `TimeStepExtractor`, expansion
  target search lived in `RawActionExecutor`, and target recovery lived in
  `ActionEffectVerifier`. The same command could therefore have a valid screen
  candidate, a different raw expansion point and a third effect target.
  Rejection memory was additionally split between agent screen coordinates,
  extractor world coordinates and expansion-anchor state.
- **Required correction:**
  1. make raw/world target the canonical placement identity;
  2. emit the validated raw target directly without translator resampling;
  3. quarantine failed target regions by action, footprint and world radius;
  4. resample from a deterministic finite candidate set;
  5. separate PySC2 rejection, missing builder order and missing structure
     effect.
- **Implemented evidence:**
  - one episode-scoped `RawPlacementService` is injected into the extractor,
    raw executor and effect verifier;
  - the service owns persistent resource observations, candidate projection,
    expansion target resolution, command-to-world-target binding and permanent
    quarantine;
  - raw pre-dispatch failure now quarantines the coordinate or anchor before
    the command is terminalized;
  - EffectVerifier reads the exact placement bound by raw dispatch rather than
    reconstructing it from screen coordinates;
  - deterministic contracts cover pre-dispatch world quarantine, permanent
    expansion-anchor suppression and effect confirmation from the shared
    target.
- **Acceptance criteria:**
  - one quarantined region is never dispatched again for that action;
  - a raw command contains requested and resolved target provenance;
  - screen reprojection cannot bypass world-space deduplication;
  - build-failure stage/code coverage remains 100%.
  - candidate, dispatched and effect-evidence world targets are identical for
    100% of tracked builds.

### SCX-PT-022: exhausted expansion search cannot discover later anchors

- **Priority:** P0
- **Status:** implemented in raw discovery; natural-terminal expansion evidence pending
- **Components:** expansion observation, scout generation, commitment, Nexus
  raw placement
- **Evidence:**
  - all three seeds remained on one Nexus;
  - generation 1 exhausted around loops 854-865 after 8/8 waypoints with
    `evaluated_anchors=[]`;
  - valid expansion anchors first appeared much later (seed 2 around loop 7,288
    and seed 1 around loop 15,871);
  - `request_immediate_progress()` returns immediately once the controller is
    exhausted, so no generation 2 is created.
- **Impact:** later map knowledge cannot reopen expansion, even when a valid
  resource cluster becomes observable.
- **Root cause:** exhaustion is an irreversible state on a controller initialized
  with `generation_id=1`. It is not a terminal of one strategic search that can
  be followed by a deliberate next generation. The feature-camera scout also
  confuses viewport coverage with world discovery.
- **Required correction:**
  1. in raw mode persist every visible neutral resource directly from raw
     observations;
  2. create a monotonic new generation when new anchors appear or all previous
     anchors are invalidated;
  3. bind one Runtime commitment to one generation;
  4. permanently suppress invalid anchors for the episode;
  5. place Nexus from the chosen resource cluster in raw coordinates;
  6. terminalize as Nexus confirmed or explicit finite candidate exhaustion.
- **Implemented evidence:**
  - raw observations persist neutral resource units without requiring the
    feature camera to visit them;
  - the extractor publishes a monotonic raw expansion generation and opens a
    new generation when later candidate anchors appear;
  - failed Nexus effect reports and immediate raw rejection permanently suppress
    the exact anchor;
  - Broker direct-decision provenance now retains command-to-anchor ownership;
  - deterministic generation-2 and permanent-suppression contracts pass. The
    bounded canary ended before a Nexus frontier was reached.
- **Acceptance criteria:**
  - a newly discovered anchor after exhaustion creates generation 2;
  - one anchor is attempted at most once after invalidation;
  - one attempt ends in a Nexus or a structured terminal;
  - at least one of seeds `[0,1,2]` builds a second Nexus or reports finite
    full-candidate exhaustion.

### SCX-PT-023: Defense was a combat router, not a defensive strategy Agent

- **Priority:** P1
- **Status:** architecture implementation complete; live emergency response pending
- **Components:** DefenseAgent, RaceProfile, Role Agents, Intent Arbiter
- **Evidence:**
  - all three raw natural-terminal games lost their original Nexus;
  - seed 2 faced Mutalisks without an anti-air production response;
  - the final 500-loop windows contained no Defense Intent despite sufficient
    banked resources in at least one terminal state.
- **Impact:** the system can classify a threat as critical while continuing the
  macro frontier unchanged, so correct Situation analysis does not become an
  executable defensive agenda.
- **Root cause:** `DefenseAgent` only routed an already available
  `Attack_Unit` or `Move_Minimap`. It could not derive race-specific production,
  static defense, anti-air prerequisite closure, resource preemption or
  last-resort worker defense. Offense, FocusFire and Retreat also inherited a
  shared Tactical source ID, so their apparent role separation did not identify
  the actual responsible Agent.
- **Required correction:** compile bounded emergency options from a typed
  RaceProfile doctrine, emit emergency resource claims before Fast Executor,
  and give every tactical proposal one concrete role-Agent owner.
- **Implemented evidence:**
  - Protoss, Terran and Zerg RaceProfiles now declare ground production,
    anti-air production, static/anti-air defense, prerequisites and worker
    defense actions;
  - Defense compiles candidate-bound emergency Intents with a common exclusion
    group, maximum urgency and real mineral/gas/supply claims;
  - emergency claims can preempt non-emergency agenda commitments in the
    existing Intent Arbiter;
  - worker defense is considered only at critical threat and only when no
    combat response exists;
  - Offense, FocusFire and Retreat now stamp distinct Agent provenance before
    strategic adaptation.
- **Acceptance criteria:**
  - critical air threat produces a legal anti-air production, prerequisite or
    anti-air static-defense Intent within 8 loops;
  - critical ground threat produces production/static/combat response within
    8 loops;
  - emergency resource claims preempt a conflicting non-emergency reservation;
  - worker defense is never emitted while a viable combat response exists;
  - role-to-intent-to-command lineage names one of the seven concrete Agents for
    100% of commands.

### SCX-PT-028: retreat durability and membership were not actor-local

- **Priority:** P1
- **Status:** implementation complete; natural-terminal acceptance pending
- **Components:** Observation contract, Tactical role Agents, RetreatAgent
- **Evidence:**
  - seed 0 repeatedly re-entered `retreat_arrived` for a Void Ray group after
    global threat had fallen;
  - `_units_for_actor()` selected all units with the same type suffix, and
    `_retreat_intents()` used the minimum `health_fraction` of that global set;
  - Protoss shield was absent from `UnitState`.
- **Impact:** one damaged Void Ray can keep another logical actor in retreat,
  block Offense target search and prevent a large surviving army from converting
  an advantage.
- **Root cause:** logical control-group identity was reconstructed from a team
  name rather than the exact raw tag membership. Durability was hull-only and
  reduced by global minimum, which is incorrect for shielded multi-unit groups.
- **Required correction:** project exact actor scopes from team raw tags, carry
  shield and combined durability, and evaluate retreat hysteresis on only the
  owning actor's members.
- **Implemented evidence:**
  - Worker snapshots include non-empty team `unit_tags`;
  - Observation mapping attaches `actor_scopes`, `shield_fraction` and
    `(health + shield) / (health_max + shield_max)` durability;
  - actor lookup prefers exact membership and only falls back to the historical
    type name when reading old journals;
  - retreat uses the actor-local median durability, preserving single-unit
    behavior without allowing one unrelated straggler to poison another actor;
  - new optional fields are omitted from legacy serialization when absent, so
    all pinned policy-corpus hashes remain unchanged.
- **Acceptance criteria:**
  - units from a different same-type actor cannot trigger retreat;
  - a shielded Protoss actor does not retreat solely because hull health is low;
  - a genuinely low-durability actor still retreats within 8 loops;
  - Retreat state becomes obsolete after recovery/arrival and does not block
    later Offense search.

### SCX-PT-025: censored episodes still influence lesson promotion

- **Priority:** P1
- **Status:** implementation complete; evolving/frozen paired acceptance pending
- **Components:** Playbook review, lesson promotion, rule promotion
- **Evidence:**
  - the executable rule database remained at 36 rules after seed 2 errored,
    showing rule mutation isolation works;
  - `playbook_lesson_promoted` was still emitted for the censored/error episode;
  - lesson count grew from 24 to 25 despite an infrastructure terminal.
- **Impact:** the executable guard is protected, but Console/reporting and
  future consolidation can still treat infrastructure failure as learned
  tactical evidence.
- **Root cause:** censored eligibility is enforced in rule consolidation but
  not at the earlier lesson-promotion boundary.
- **Required correction:**
  1. keep censored cases and lessons queryable for diagnostics;
  2. mark them `promotion_eligible=false`;
  3. exclude them from lesson promotion and rule support;
  4. record the exclusion in report/Console lineage;
  5. rerun frozen/evolving comparison only after raw execution gates pass.
- **Implemented evidence:** ERROR and TRUNCATED reviews are marked ineligible
  before lesson consolidation; they generate neither promoted lessons nor rule
  support. Regression tests cover both terminals. A fresh paired experiment is
  still required before claiming Playbook quality improvement.
- **Acceptance criteria:**
  - censored/error episodes create zero promoted lessons and zero executable
    support;
  - diagnostics retain source run, seed and terminal reason;
  - uncensored promotion conservation remains 100%;
  - Playbook quality is not claimed without a valid paired experiment.

## Implementation order

1. **Done:** add the raw action-space configuration and direct Runtime decision
   path.
2. **Done:** implement exact raw actor/source resolution and Protoss action
   mapping.
3. **Done:** bind EffectVerifier preparation to the same tags and raw/world
   target.
4. **Done:** remove camera/selection/translator/abort from the raw critical
   path.
5. **Done:** correct Move order semantics and raw/world placement quarantine.
6. **Done:** make expansion discovery and generation lifecycle raw-observation
   driven.
7. **Done:** exclude censored lessons at the promotion boundary.
8. **Done:** run deterministic contracts and a bounded live canary.
9. **Done:** add RaceProfile emergency doctrine and concrete tactical Agent
   ownership.
10. **Done:** make `RawPlacementService` the sole world-placement authority.
11. **Done:** add shield-aware exact actor membership and the `Void Ray` parser
   alias.
12. **Pending:** run Protoss seeds `[0,1,2]` to natural terminal with active
   Arbiter and frozen Playbook, then close only the issues whose long-run
   criteria are observed.

## Required engineering gates

```text
uv run pytest
uv run ruff check src tests integrations/llm_pysc2/src
uv run mypy
```

Worker Python 3.9 checks must additionally cover:

- simultaneous RAW actions plus feature/RGB observations;
- exact actor, builder and producer tag binding;
- every supported Protoss raw function and argument contract;
- PySC2 acceptance settlement on the following observation;
- order 547 movement acquisition and true arrival;
- world-space build quarantine and deterministic resampling;
- expansion generation 2 after a later raw anchor discovery;
- censored lesson-promotion exclusion;
- feature mode compatibility.

Passing unit tests does not close a P0 live issue. The raw path first requires a
2,000-3,000-loop canary, followed by Protoss seeds `[0,1,2]` natural terminal.
