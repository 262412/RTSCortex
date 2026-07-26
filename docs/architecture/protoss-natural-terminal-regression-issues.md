# Protoss natural-terminal regression issue register

Status date: 2026-07-25

This register contains only issues that remain open after:

```text
protoss-frozen-natural-terminal-postfix-20260725T103903Z
```

The run used HIMA Protoss a/b/c, active Strategic Intent Arbiter, a frozen
active Playbook, `Simple64`, Protoss versus VeryEasy Zerg, and
`game_steps_per_episode: 0`.

| Seed | Outcome | Steps | Meaningful success | Build | Production |
|---|---:|---:|---:|---:|---:|
| 0 | defeat | 16,490 | 47/75 | 9/14 | 25/25 |
| 1 | defeat | 15,814 | 25/59 | 7/37 | 18/18 |
| 2 | error | 9,060 | 53/92 | 11/12 | 28/28 |

Production provenance, terminal-report exactly-once, duplicate-dispatch
protection, candidate-domain validation, friendly-target safety and threat
classification remained healthy. Resolved issues are not retained in this
active register.

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

The repeated failures have one common architectural cause. The current live
path contains two runtimes:

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

### SCX-PT-026: dual execution ownership causes non-local failures

- **Priority:** P0
- **Status:** implemented; bounded live canary passed, natural-terminal acceptance pending
- **Components:** Worker, Fast Executor, Bridge, PySC2 action space
- **Evidence:**
  - seed 2 ended with
    `BridgeIntegrityError: upstream aborted an action with no unique command:
    CombatGroup7/Adept-1/Attack_Unit`;
  - previous runs alternated between camera budget exhaustion, stale selection,
    producer provenance loss, `cannot find unit on screen`, and unattributed
    aborts although the Runtime command was valid;
  - the current Worker calls `MainAgent.step()` after RTSCortex validation, so
    an independent mutable state machine still decides the final primitive.
- **Impact:** execution correctness depends on UI state that is unrelated to
  semantic legality. One stale upstream queue can terminate a natural-terminal
  episode and can contaminate Playbook evidence.
- **Root cause:** RTSCortex owns command intent and tracking while LLM-PySC2
  owns actor resolution and final dispatch. There is no single authoritative
  command-to-SC2 transition.
- **Required correction:**
  1. add an explicit `raw` execution action-space setting;
  2. keep feature/RGB/raw observations enabled;
  3. bind commands to living raw unit tags after Runtime validation;
  4. translate supported Protoss semantic actions to exactly one raw primitive;
  5. record that primitive directly under the command ID;
  6. settle PySC2 acceptance on the following observation;
  7. bypass `MainAgent.step()`, camera, selection and upstream translator in raw
     mode;
  8. retain feature mode only as an explicit compatibility path.
- **Implemented evidence:**
  - `execution_action_space: raw` starts SC2 with both raw and feature/RGB
    interfaces;
  - `RawActionExecutor` binds the exact actor, builder or producer tags and emits
    one final raw primitive per command;
  - raw Worker ticks call the RTSCortex decision broker directly and never call
    `MainAgent.step()`;
  - patch `0023` makes PySC2's diagnostic action printer tolerate the absence of
    feature-only `available_actions`;
  - the bounded canary completed 2,000 observations and 14/14 meaningful
    commands without an upstream action abort.
- **Acceptance criteria:**
  - raw mode emits no orchestration camera or selection primitives;
  - upstream aborts and `cannot find unit on screen` cannot terminate raw runs;
  - every raw primitive has exactly one command ID;
  - actor tags in effect evidence equal the tags sent to PySC2;
  - supported Protoss actions have 100% raw mapping coverage;
  - feature/RGB Live Console observations remain available.

### SCX-PT-016: movement was executed but falsely classified as failed

- **Priority:** P0
- **Status:** implementation complete; long-run combat evidence pending
- **Components:** RawActionBridge, MoveVerifier, Offense/Defense navigation
- **Evidence:**
  - the three runs produced 44 meaningful Move commands but only one success;
  - 40 failures were `move_order_not_observed`;
  - failed evidence repeatedly contained `worker_orders=["547"]` and observable
    displacement while the verifier required order `13`.
- **Impact:** real movement is reported as failure, causing actor-local
  navigation, Defense cooldown, Arbiter feedback and Playbook learning to
  re-arm or punish actions that SC2 actually executed.
- **Root cause:** PySC2 raw order projection generalizes SC2 ability 16 to
  `RAW_FUNCTIONS.Move_Move_pt` function ID 547. The verifier used the generic
  smart-move function ID 13 (`Move_pt`, ability 3794). In feature mode, selected
  actor identity could also differ from configured team identity.
- **Required correction:**
  1. dispatch `Move_Move_pt` with the exact living actor tags;
  2. verify raw order 547, not 13;
  3. compute arrival from those same surviving tags;
  4. keep one actor-local waypoint lifecycle until effect terminal;
  5. distinguish order acquisition, travel, arrival, actor loss and timeout.
- **Implemented evidence:**
  - raw movement emits `Move_Move_pt` function/order 547 with the exact living
    team tag set;
  - MoveVerifier accepts both legacy feature evidence and raw order 547, and raw
    mode measures the same world-coordinate actor centroid used at dispatch;
  - deterministic verifier and RawActionExecutor contracts pass. The bounded
    canary did not reach combat movement, so this issue is not closed yet.
- **Acceptance criteria:**
  - order 547 is recognized on deterministic and live observations;
  - unrelated movement orders cannot confirm the command;
  - actor tags are identical at dispatch and verification;
  - accepted movement with real displacement is not failed at 16 loops;
  - true-arrival rate is at least 80% in deterministic contracts.

### SCX-PT-027: build placement retries are screen-local, not world-stable

- **Priority:** P0
- **Status:** implemented; failure-path live evidence pending
- **Components:** build candidate generation, RawActionBridge, BuildVerifier
- **Evidence:**
  - seed 1 confirmed only 7 of 37 build commands;
  - repeated failures targeted the same raw/world region around
    `[17.9375, 28.625]` through different screen coordinates;
  - failures included both `no_build_order_observed` and
    `target_not_created`.
- **Impact:** a bad location can be retried indefinitely under different camera
  projections, consuming the Builder, resources and strategic frontier.
- **Root cause:** rejection memory is keyed primarily by feature-screen
  position. Camera reprojection makes one world location appear as multiple
  screen positions, and the translator performs another placement decision
  after Runtime candidate validation.
- **Required correction:**
  1. make raw/world target the canonical placement identity;
  2. emit the validated raw target directly without translator resampling;
  3. quarantine failed target regions by action, footprint and world radius;
  4. resample from a deterministic finite candidate set;
  5. separate PySC2 rejection, missing builder order and missing structure
     effect.
- **Implemented evidence:**
  - `screen_world_target` is now sent directly to the raw build function;
  - immediate PySC2 rejection and later EffectVerifier failure both quarantine
    the world target;
  - future candidate projection filters a footprint-sized radius around that
    world target, so camera-relative coordinates cannot evade deduplication;
  - bounded canary build confirmation was 6/6 and produced no retries. A live
    intentional-failure case or natural-terminal failure is still required to
    close the rejection-path criterion.
- **Acceptance criteria:**
  - one quarantined region is never dispatched again for that action;
  - a raw command contains requested and resolved target provenance;
  - screen reprojection cannot bypass world-space deduplication;
  - build-failure stage/code coverage remains 100%.

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

### SCX-PT-023: Defense retries are amplified by false movement feedback

- **Priority:** P1
- **Status:** raw feedback dependency fixed; long-run Defense behavior pending
- **Components:** DefenseAgent, actor-local state, MoveVerifier, Arbiter
- **Evidence:**
  - Defense intent volume fell to 26/2/52 after actor-local hysteresis;
  - most remaining failed responses were movement failures;
  - the same observations contain move order 547, proving that false verifier
    feedback can still trigger cooldown/retry behavior.
- **Impact:** Defense can occupy the agenda or back off from a valid response
  based on incorrect execution feedback.
- **Root cause:** the Agent state machine is now actor-local, but its terminal
  input still comes from the incorrect Move order contract.
- **Required correction:** complete SCX-PT-016, then tune Defense only from raw
  dispatch/effect evidence.
- **Implemented evidence:** raw commands can no longer receive feature-selection
  feedback, and raw movement uses the same actor tags in dispatch and effect
  preparation. No threat occurred in the bounded canary, so retry/hysteresis
  behavior remains a natural-terminal gate.
- **Acceptance criteria:**
  - one actor/target signature has no overlapping active response;
  - valid movement does not enter failure cooldown;
  - emergency response remains within 8 loops;
  - retries are bounded by real terminals, not observation count.

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
9. **Pending:** run Protoss seeds `[0,1,2]` to natural terminal with active
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
