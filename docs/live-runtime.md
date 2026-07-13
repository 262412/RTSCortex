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
  - `pvz_task1_level1`: `Maps/llm_pysc2/pvz_task1_level1.SC2Map`, SC2 build
    92440 (5.0.13) or newer;
  - `2s3z`: `Maps/llm_smac/2s3z.SC2Map`, compatible with the official Linux
    SC2 4.10/Base75689 package;
- both reviewed patches from `integrations/llm_pysc2/patches` applied to the pinned
  LLM-PySC2 checkout (asynchronous waiting and deterministic SC2 seeding).

The runtime never downloads StarCraft II or applies upstream patches automatically. Keep
the pinned submodule clean between live runs. To apply the reviewed patches explicitly
for a live session:

```bash
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
```

After the live session, use the reverse commands documented in
`integrations/llm_pysc2/patches/README.md` to return the submodule to its pinned, clean
state.

The Blizzard download page currently exposes Linux packages only through SC2 4.10
(Base75689). That binary can run older maps, but the v0.1 PvZ maps were saved by build
92440; a real create-game test shows Base75689 crashes while loading them. `doctor` and
the live preflight therefore reject older builds instead of relying on path checks alone.

Validate and launch the Linux-compatible `2s3z` bridge with:

```bash
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_2s3z.yaml \
  --require-sc2
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_2s3z.yaml
```

The default `live_2s3z.yaml` uses the deterministic Fake Provider, so it tests the real
SC2 process, observation mapping, Unix-socket API, typed planning, actor routing, upstream
translator, PySC2 execution feedback, and cleanup without requiring a model endpoint. A
run that reaches `max_agent_steps` is recorded as `truncated`; that is expected for a
bounded smoke and is distinct from a worker error.

To use the Qwen3-8B vLLM service already running on the current compute node, use the
separate real-model configuration:

```bash
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_2s3z_qwen3_8b.yaml \
  --require-sc2
SC2PATH=~/scratch/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_2s3z_qwen3_8b.yaml
```

That configuration targets `http://127.0.0.1:8000/v1`, requires the served model ID
`Qwen/Qwen3-8B`, disables Qwen thinking, limits each structured completion to 512 tokens,
and uses a 128-game-loop planning cadence. `doctor --config` checks the endpoint and exact
model ID before starting SC2. Because the address is loopback-only, the runtime must run
on the same node as vLLM. Set `RTSCORTEX_LLM_API_KEY` only when the selected
OpenAI-compatible endpoint requires authentication.

The 2s3z simulation can otherwise finish in a few wall-clock seconds while a local model
is still planning. While `ActionBatch.planner_pending` is true, the Qwen configuration
therefore delays idle/no-op SC2 steps by 0.75 seconds. Attack, camera, selection, and other
real primitives are never delayed, so cached plans and reflex actions retain the fast
path. This fixed delay is tuned to the current node's observed Qwen latency; slower model
deployments should measure and adjust it. Fake-provider configs leave the delay at zero.

The model-facing context keeps the typed observation, alerts, available actions, and
compact decision/execution memory while omitting the verbose raw observation text. The
complete observation is still retained in `events.jsonl`; this keeps the live Qwen
request below its 4096-token context limit without reducing run traceability.

Asynchronous plans record their source loop, acceptance loop, and age. Their TTL begins
at acceptance, while `Attack_Unit` commands carry a `unit_exists` precondition so a target
that disappeared during model inference is rejected before reaching the bridge.

The `2s3z` arena is only 32 by 32 world units. Its camera is clamped at the map edge, so
an initial unit cannot satisfy LLM-PySC2's large-map centering threshold. The bridge
therefore marks strict centering complete for this scenario; upstream still infers the
coordinate range from the first observation before entering its grouping and decision
loop. The PvZ scenario continues to use the complete upstream camera calibration.

Validate and launch the original Protoss PvZ scenario only with a compatible SC2 build:

```bash
SC2PATH=/path/to/StarCraftII \
  uv run rtscortex doctor \
  --config configs/experiments/live_pvz.yaml \
  --require-sc2
SC2PATH=/path/to/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_pvz.yaml
```

The worker receives `RTSCORTEX_RUNTIME_SOCKET`, `RTSCORTEX_RUN_ID`,
`RTSCORTEX_EPISODE_ID`, `RTSCORTEX_SCENARIO`, `RTSCORTEX_SEED`, and `SC2PATH`. The current
bridge also receives `RTSCORTEX_SOCKET` as its socket variable. Worker stdout and stderr,
the config snapshot, SQLite state, and the JSONL event journal are stored in the run's
artifact directory.

Live runs currently pass `--save_replay=false` to PySC2 and do not create
`.SC2Replay` files. Use `rtscortex report <run-dir>` to turn the JSONL journal into a
readable `timeline.md` until replay capture is deliberately enabled in a later version.

## Failure behavior

The worker starts only after the runtime health endpoint responds. A non-zero worker exit
causes the CLI to exit with status 1 after recording an error result. A clean worker exit
without `/v1/episode/end` records a truncated result. Cancellation terminates the complete
worker process group before the API and runtime store are closed.
