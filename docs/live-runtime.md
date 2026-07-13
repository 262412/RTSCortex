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
- SC2 build 92440 (5.0.13) or newer for the fixed `pvz_task1_level1` map;
- `Maps/llm_pysc2/pvz_task1_level1.SC2Map` below `SC2PATH`;
- both reviewed patches from `integrations/llm_pysc2/patches` applied to the pinned
  LLM-PySC2 checkout (asynchronous waiting and deterministic SC2 seeding).

The runtime never downloads StarCraft II or applies upstream patches automatically. To
apply the reviewed patches explicitly:

```bash
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
```

The Blizzard download page currently exposes Linux packages only through SC2 4.10
(Base75689). That binary can run older maps, but the v0.1 PvZ maps were saved by build
92440; a real create-game test shows Base75689 crashes while loading them. `doctor` and
the live preflight therefore reject older builds instead of relying on path checks alone.

Validate and launch the fixed Protoss PvZ scenario with:

```bash
SC2PATH=/path/to/StarCraftII uv run rtscortex doctor --require-sc2
SC2PATH=/path/to/StarCraftII \
  uv run rtscortex run --config configs/experiments/live_pvz.yaml
```

The worker receives `RTSCORTEX_RUNTIME_SOCKET`, `RTSCORTEX_RUN_ID`,
`RTSCORTEX_EPISODE_ID`, `RTSCORTEX_SCENARIO`, `RTSCORTEX_SEED`, and `SC2PATH`. The current
bridge also receives `RTSCORTEX_SOCKET` as its socket variable. Worker stdout and stderr,
the config snapshot, SQLite state, and the JSONL event journal are stored in the run's
artifact directory.

## Failure behavior

The worker starts only after the runtime health endpoint responds. A non-zero worker exit
causes the CLI to exit with status 1 after recording an error result. A clean worker exit
without `/v1/episode/end` records a truncated result. Cancellation terminates the complete
worker process group before the API and runtime store are closed.
