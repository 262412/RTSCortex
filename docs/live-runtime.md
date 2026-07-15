# Live LLM-PySC2 runtime

Live episodes use two isolated processes: the Python 3.11 RTSCortex runtime and the pinned
LLM-PySC2 worker in Python 3.9. `rtscortex run` owns both processes and uses a unique Unix
socket for each episode.

## Required installation

Before a live run, provide all of the following:

- a Python 3.9 worker at
  `~/fastscratch/envs/rtscortex-llm-pysc2/bin/python`, or set
  `RTSCORTEX_LLM_PYSC2_PYTHON`;
- an executable `SC2_x64` below `SC2PATH`;
- a supported scenario and its matching map/build combination:
  - `Simple64`: `Maps/Melee/Simple64.SC2Map`, compatible with the official Linux
    SC2 4.10/Base75689 package and recommended for Ladder/Melee live smoke runs;
  - `pvz_task1_level1`: `Maps/llm_pysc2/pvz_task1_level1.SC2Map`, SC2 build
    92440 (5.0.13) or newer;
  - `2s3z`: `Maps/llm_smac/2s3z.SC2Map`, compatible with the official Linux
    SC2 4.10/Base75689 package, retained as a legacy task map;
- all nine reviewed patches from `integrations/llm_pysc2/patches` applied to the pinned
  LLM-PySC2 checkout (asynchronous waiting, deterministic SC2 seeding, row-major
  build-coordinate validation, structured primitive provenance, safe Near placement, and
  explicit pre-translation abort reporting, transient gas/transport unit grace, and Nexus
  resource-clearance validation with exact screen-to-world scaling and visibility checks).

The runtime never downloads StarCraft II or applies upstream patches automatically. Keep
the pinned submodule clean between live runs. To apply the reviewed patches explicitly
for a live session:

```bash
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0003-fix-build-feature-plane-coordinate-order.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0004-expose-structured-translation-results.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0005-fix-near-base-placement.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0006-report-pretranslation-action-aborts.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0007-preserve-transient-team-units.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0008-enforce-nexus-resource-clearance.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0009-use-exact-nexus-screen-scale.patch
```

After the live session, use the reverse commands documented in
`integrations/llm_pysc2/patches/README.md` to return the submodule to its pinned, clean
state.

The Blizzard download page currently exposes Linux packages only through SC2 4.10
(Base75689). That binary can run older maps, but the v0.1 PvZ maps were saved by build
92440; a real create-game test shows Base75689 crashes while loading them. `doctor` and
the live preflight therefore reject older builds instead of relying on path checks alone.

## Recommended Simple64 Ladder/Melee smoke

`Simple64` is the recommended compute-center live path. It uses the RTSCortex-owned
minimal Protoss melee configuration for building, production, Zealot, and Stalker
control while leaving SC2 lifecycle and PySC2 execution in LLM-PySC2. The opponent is a
built-in VeryEasy Zerg bot using the Macro build.
Production and research actions are exposed only when their completed idle source, mineral,
vespene, supply, and prerequisite requirements are all currently satisfied; this includes the
50-mineral/50-vespene Warp Gate research cost.

Validate and launch the deterministic Fake-provider smoke with:

```bash
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_simple64.yaml \
  --require-sc2
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_simple64.yaml
```

`live_simple64.yaml` uses the deterministic Fake Provider, so it tests the real
SC2 process, observation mapping, Unix-socket API, typed planning, actor routing, upstream
translator, PySC2 execution feedback, and cleanup without requiring a model endpoint. A
run is capped at 224 game loops and 256 agent steps.

Use the separate real-model configuration for the Qwen3-8B vLLM service:

```bash
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_simple64_qwen3_8b.yaml \
  --require-sc2
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_simple64_qwen3_8b.yaml
```

For deterministic multi-seed regression, `run` accepts a non-negative override without editing
the checked-in configuration, for example `--seed 0`, `--seed 1`, and `--seed 2`. The override is
written into the run's config snapshot and forwarded to SC2Env.

That configuration targets `http://127.0.0.1:8000/v1`, requires the served model ID
`Qwen/Qwen3-8B`, disables Qwen thinking, limits each structured completion to 256 tokens,
caps the combined system and user prompt at 9,000 characters, and uses a 112-game-loop
planning cadence. `doctor --config` checks the endpoint and exact model ID before starting
SC2. Because the address is loopback-only, the runtime must run on the same node as vLLM.
Set `RTSCORTEX_LLM_API_KEY` only when the selected OpenAI-compatible endpoint requires
authentication. The short configuration is capped at 1024 game loops and 1120 agent
steps. `live_simple64_smoke_qwen3_8b.yaml` runs 3,000 game loops before the complete-match
configuration is used.

### Simulation and planning timing

Both recommended configurations use `step_mul=1`: each environment step advances exactly
one SC2 game loop. `simulation_speed_multiplier=0.25` applies a fixed monotonic clock to
every step, producing 5.6 game loops per wall-clock second from SC2's 22.4-loop baseline.
If work misses a wall-clock deadline, the clock skips obsolete deadlines rather than
bursting through catch-up steps; it never skips SC2 game loops.

`pause_until_first_plan=true` prevents the game from advancing until the first valid plan
has completed. The pacing clock starts fresh when that one-time barrier releases. Later
planner cycles are asynchronous: the SC2 loop continues at the fixed rate while the
runtime validates cached actions and evaluates reflexes. A first-plan timeout or error is
a startup failure rather than permission to begin with an empty plan.

These limits make both configurations live integration smokes, not complete ladder
matches or win-rate measurements. When the SC2 episode reaches its configured game-loop
cap before either player wins, the current worker reports a zero-reward `draw`; that
bounded result must not be interpreted as a completed match. Live execution always
passes `--save_replay=false`.

### Model context budget and command economy

Model context is rebuilt from typed state on every reflection or planning call; it is not
an ever-growing transcript. The structural compactor retains current economy, production,
alerts, available actions, the active plan, the latest execution, reusable lessons, and
actionable spatial coordinates. It groups large own-unit, structure, and visible-enemy
lists by type, coalesces repeated events, and then removes the oldest bounded history until
the complete system-plus-user prompt fits `context.max_prompt_chars`. If the mandatory
current state cannot fit, the call fails explicitly instead of sending an oversized or
truncated JSON prompt.

The Qwen configuration uses a 9,000-character prompt ceiling, at most eight recent events,
six lessons, and one episode summary. This is a conservative envelope for the current
structured English/JSON prompts served through the 4,096-token Qwen endpoint, with a
separate 256-token completion limit. The timeline records the configured budget, original
and final character counts, and every dropped or aggregated category for each model call.

Planning asks for at most two short plan steps and three typed actions. The model must emit
only actions needed in the current decision cycle, with one action per actor, and must not
repeat a successful action or an active-plan command that remains valid. Full observations
remain in `events.jsonl`, so reducing model input does not reduce experiment traceability.

### Deterministic goal progress and policy shadowing

For accepted plans that contain supported Protoss state-changing actions, Runtime creates a
typed `GoalSpec` and evaluates it on every observation with `GoalProgressVerifier`. The
result records completed and missing requirements, deterministic blockers, every action
that can advance the goal now, and `unique_next_action` only when exactly one such action
exists. Resource totals, production queues, completed/in-progress structures, units,
upgrades, prerequisites, action availability, and defensive alerts are state evidence;
the LLM does not decide whether its own goal advanced.

The same `GoalProgressReport` is supplied to Reflection and Planning and emitted as a
durable `goal_progress` event for the Live Console. `ProgressGuard` removes `Stop`,
`Hold_Position`, and semantic no-op commands while a goal can advance or its effect is
already in progress, unless the observation explicitly requires a defensive hold. An
empty follow-up plan retains the last measurable goal, so waiting for a build effect does
not erase its progress state.

Policy candidates can be evaluated without giving them dispatch access:

```bash
uv run rtscortex policy-shadow \
  ~/scratch/outputs/RTSCortex/<run-directory> \
  --limit 11 --stride 6
```

Every candidate receives the same historical observations, fixed Protoss opening goal,
and deterministic progress reports. The current OpenAI-compatible Qwen configuration is
loaded from the run snapshot. HIMA Protoss-a/b/c and HierNet-SC2 remain explicit
`unavailable` entries until local weights and RTSCortex action adapters are configured;
this command never downloads them. Use `--no-current-qwen` to validate the fixture and
report path entirely offline. The resulting `policy-shadow-comparison.json` separates
schema legality, goal-advancing choices on actionable fixtures, control-action violations,
latency, model failures, and unavailable candidates.

### Build position handling

The bridge derives one structured candidate domain for every exposed target or position
action. Screen-build candidates use the current buildability, pathing, ownership, and
power layers. Their complete footprint must be buildable, pathable, powered when required,
and empty (`player_relative == 0`) in PySC2's row-major feature planes. Feature-unit radii
are dilated around the footprint, so a Probe or existing structure cannot overlap its edge.
The model schema and Runtime membership validator consume exactly this same candidate set;
the candidates are not duplicated into free-form observation text.

Pylon, Gateway, and Cybernetics Core use screen candidates with their appropriate footprint,
power, resource, and prerequisite rules. Assimilator accepts only an unoccupied, visible
neutral geyser near a completed Nexus. Builder `Move_Minimap` candidates expose unseen remote
neutral minimap clusters ordered by distance from the current base. Deterministic map-spanning
pathable points are used only when no unseen resource cluster is available. This gives the Planner
an explicit scouting path before a remote expansion is eligible for construction; the Worker
revalidates the selected minimap point immediately before dispatch.
Nexus accepts a unique anchor only after at least five
mineral patches in its resource cluster have been scouted and are currently visible; this avoids
issuing a 400-mineral order into an unverified fogged expansion. The cluster must have no existing
Protoss, Terran, or Zerg townhall belonging to either player. Arbitrary Nexus and Assimilator
screen placement is not exposed. Immediately
before translation, the Worker reprojects the candidate's Bridge-private world target into
the current camera and recomputes the same semantics. A stale screen coordinate may move only
to the nearest equivalent candidate within two sampling strides of that reprojection; tag
targets are never substituted. Camera provenance never enters the public v1.1 payload.
`Move_Screen` and `Ability_Blink_Screen` use the same world-target provenance, then re-enter
the current pathable candidate domain before dispatch. The Builder's pre-Gateway movement also
rechecks the current power mask. Reports retain the Planner's requested pixel coordinate and
record the reprojected coordinate separately in `resolved_arguments`.

Nexus translation moves the camera to the exact resource cluster, emits one translator-owned
transport no-op, and resolves the final 5x5 placement on the following observation. This
settlement tick is required because SC2 can update the camera transform one observation before
its raw `is_on_screen` flags; projected coordinates outside the feature screen are never used.
The final candidate must also keep every mineral 6–9 world units and every geyser 7–10 world
units from the Nexus center. Candidates are scored by their distance from the ideal resource
ring before centroid proximity, preventing a nominally buildable 5x5 footprint from landing
inside the mineral line. These distances use the exact floating-point `screen_size / 24`
conversion; the final anchor, candidate center, and complete footprint must be visible.
RTSCortex repeats the complete-footprint visibility check against the translator's resolved
screen position before allowing the final Nexus primitive to enter PySC2.

Build execution now uses two-phase confirmation. First, the next SC2 observation must show
that PySC2 accepted the primitive. The `ActionEffectVerifier` then defers the final
`ExecutionReport` while it checks raw observations for a new structure tag of the expected
type at the resolved command target. Assimilator keeps the exact requested geyser as that
target; Nexus uses the translator's actual final screen placement rather than the resource
anchor. This target-matched new tag is the only success criterion. Mineral spending and the
selected Builder's status and ability orders remain diagnostic evidence; unrelated spending,
another build elsewhere, or a concurrent worker order cannot produce a false success.
Concurrent structures are matched one-to-one. If the target structure does not appear within
`environment.action_effect_timeout_game_loops` (112 loops in the Qwen configuration), the
command is normally reported as failed rather than as a false success. PySC2 exposes unit
orders as RAW function IDs; when the expected build order is still active at that deadline,
the verifier keeps the command pending for a bounded maximum. Ordinary structures use four
times the base timeout; Nexus uses twelve times because a Probe can traverse most of a melee
map before warp-in starts. The order still is not proof of success, and every wait remains
bounded. When an observed build order disappears, the verifier gives the target structure a
final 32-loop visibility grace period before classifying the command as failed; the hard timeout
still takes precedence.

The failure reason includes before/after structure count and progress, minerals, Builder
selection, status, RAW function order IDs, elapsed loops, and the effective deadline. An
order is classified as replaced only if the expected build order was observed first and a
different non-empty order appeared later; the report does not attribute that change to a
specific subsystem without matching evidence.

`Move_Minimap` uses a separate bounded confirmation rule because its argument is a feature-
minimap pixel, not a world coordinate or camera target. After PySC2 accepts the primitive, the
verifier confirms that the addressed unit either receives RAW `Move_pt` order 13 or moves at
least one world unit from its dispatch position. The global camera mask is diagnostic only and
cannot confirm a team-owned movement command. No order and no displacement within one base
effect window produces `effect_timeout`.

Asynchronous plans record independent Planner start and acceptance loops. Planner starts use
a fixed single-flight cadence; accepting a plan does not reset that cadence. Runtime-owned
command TTL begins at acceptance. Each command then moves once through pending, deferred,
dispatched, and one terminal state. At most one command per actor can remain dispatched;
later Planner commands are deferred and Reflex commands suppressed until the execution report
closes that actor. The dedicated Builder is excluded from upstream idle-worker reassignment,
and from the moment a build is tracked until it terminates the Worker disables both automatic
worker-management paths so they cannot replace the Probe's build order. An `Attack_Unit` target
must remain in the current enemy-only candidate domain; a
stale or friendly tag is rejected before reaching the Bridge.

Upstream team membership is not changed by a single camera-dependent observation. Gas and
transport occupants can temporarily disappear from both raw and feature views; the Worker
preserves their membership and waits for the upstream 40-step confirmed-death threshold. Only
confirmed disappearance removes the unit and emits a command-owned
`pre_dispatch/actor_not_available` report, so a transient observation gap cannot silently drop
or duplicate a command. Death confirmation advances on every SC2 observation even while the
upstream action loop is locked, preventing a permanently missing team head from stalling the
environment worker.
An actor disappearing while its upstream team is executing transport `No_Operation` clears that
control action without producing an execution report; unattributed semantic actions remain a
fatal Bridge integrity violation.

Live payloads use protocol 1.1. Empty decisions carry an `idle_reason` and no semantic
NoOp command. SC2 transport NoOps are counted separately and never enter gameplay success
rates. Execution reports include the action, actor, source, requested and resolved arguments,
stage, stable failure code, primitive sequence, and effect evidence. Historical protocol 1.0
journals remain readable by `report` and `replay`.

## Legacy task-map scenarios

### 2s3z micro task

`2s3z` remains available for regression coverage of the original task-map bridge. It is a
small fixed micro-combat task, not a Ladder/Melee game, and is no longer the recommended
compute-center live scenario.

```bash
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_2s3z.yaml \
  --require-sc2
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_2s3z.yaml
```

The default `live_2s3z.yaml` uses the Fake Provider. The legacy Qwen configuration uses
the same loopback `Qwen/Qwen3-8B` endpoint:

```bash
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_2s3z_qwen3_8b.yaml \
  --require-sc2
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_2s3z_qwen3_8b.yaml
```

Because this task can otherwise finish in a few wall-clock seconds, that older
configuration delays idle/no-op steps by 0.75 seconds while
`ActionBatch.planner_pending` is true. This content-dependent delay is retained for
legacy regression behavior; the recommended Simple64 configurations use the fixed game
clock instead.

The `2s3z` arena is only 32 by 32 world units. Its camera is clamped at the map edge, so
an initial unit cannot satisfy LLM-PySC2's large-map centering threshold. The bridge marks
strict centering complete for this scenario while still inferring its coordinate range
before grouping and decision execution.

### Original PvZ task

Validate and launch the original Protoss PvZ scenario only with a compatible SC2 build:

```bash
SC2PATH=/path/to/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_pvz.yaml \
  --require-sc2
SC2PATH=/path/to/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_pvz.yaml
```

This remains a task-map integration path rather than the recommended melee smoke. It
continues to use complete upstream camera calibration and requires SC2 build 92440 or
newer because Base75689 cannot load the saved map.

The worker receives `RTSCORTEX_RUNTIME_SOCKET`, `RTSCORTEX_RUN_ID`,
`RTSCORTEX_EPISODE_ID`, `RTSCORTEX_SCENARIO`, `RTSCORTEX_SEED`, and `SC2PATH`. The current
bridge also receives `RTSCORTEX_SOCKET` as its socket variable and
`RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS` from the typed environment configuration.
Worker stdout and stderr, the config snapshot, SQLite state, and the JSONL event journal
are stored in the run's artifact directory. At run shutdown, RTSCortex also derives
`timeline.md` and `summary.json` whenever the journal contains events. The JSON summary
contains per-episode metrics, classification conservation, dispatched-command terminal
coverage, and hard acceptance gate results. A report-generation warning never replaces the
worker's original success or failure status.

Every live run passes `--save_replay=false` to PySC2 and does not create
`.SC2Replay` files. Use `rtscortex report <run-dir>` to regenerate the readable
`timeline.md` and machine-readable `summary.json` until replay capture is deliberately
enabled in a later version.

## Failure behavior

The worker starts only after the runtime health endpoint responds. A non-zero worker exit
causes the CLI to exit with status 1 after recording an error result. A clean worker exit
without `/v1/episode/end` records a truncated result. Cancellation terminates the complete
worker process group before the API and runtime store are closed.
