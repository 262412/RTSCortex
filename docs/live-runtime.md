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
- all three reviewed patches from `integrations/llm_pysc2/patches` applied to the pinned
  LLM-PySC2 checkout (asynchronous waiting, deterministic SC2 seeding, and row-major
  build-coordinate validation).

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

That configuration targets `http://127.0.0.1:8000/v1`, requires the served model ID
`Qwen/Qwen3-8B`, disables Qwen thinking, limits each structured completion to 256 tokens,
caps the combined system and user prompt at 9,000 characters, and uses a 112-game-loop
planning cadence. `doctor --config` checks the endpoint and exact model ID before starting
SC2. Because the address is loopback-only, the runtime must run on the same node as vLLM.
Set `RTSCORTEX_LLM_API_KEY` only when the selected OpenAI-compatible endpoint requires
authentication. This smoke is capped at 1024 game loops and 1120 agent steps.

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

### Build position handling

The bridge derives `Build_Pylon_Screen` and `Build_Gateway_Screen` candidates from the
current feature-screen buildability, pathing, ownership, and power layers. Candidates
must have a complete build footprint that is buildable, pathable, and empty
(`player_relative == 0`) in PySC2's row-major feature planes. The reviewed upstream
coordinate patch makes LLM-PySC2 validate the same planes as `[y][x]`. On-screen
feature-unit positions are also excluded,
so a Probe or an existing structure cannot occupy a candidate footprint. Compact candidate
lines are included in the model's spatial context. Immediately before the upstream
translator executes a build action, the worker derives candidates again from the latest
observation and refreshes the screen coordinate. The coordinate in an older plan is
therefore not replayed blindly after camera, units, or map state changes.

Build execution now uses two-phase confirmation. First, the next SC2 observation must show
that PySC2 accepted the primitive. The `ActionEffectVerifier` then defers the final
`ExecutionReport` while it checks raw observations for the target structure count and
build progress, mineral spending, and the selected Builder's tag, status, and ability
orders. A newly visible target structure confirms the action; mineral spending together
with the expected build ability on that Builder also confirms it before the structure is
visible. If neither condition appears within
`environment.action_effect_timeout_game_loops` (112 loops in the Qwen configuration), the
command is reported as failed rather than as a false success.

The failure reason includes before/after structure count and progress, minerals, Builder
selection, status, and order IDs. Those diagnostics make it possible to distinguish a
missing or lost worker selection, an order changed by worker management, and a
feature-action placement that was accepted by the API but produced no construction.

Asynchronous plans record their source loop, acceptance loop, and age. Their TTL begins
at acceptance, while `Attack_Unit` commands carry a `unit_exists` precondition so a target
that disappeared during model inference is rejected before reaching the bridge.

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
are stored in the run's artifact directory.

Every live run passes `--save_replay=false` to PySC2 and does not create
`.SC2Replay` files. Use `rtscortex report <run-dir>` to turn the JSONL journal into a
readable `timeline.md` until replay capture is deliberately enabled in a later version.

## Failure behavior

The worker starts only after the runtime health endpoint responds. A non-zero worker exit
causes the CLI to exit with status 1 after recording an error result. A clean worker exit
without `/v1/episode/end` records a truncated result. Cancellation terminates the complete
worker process group before the API and runtime store are closed.
