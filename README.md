# RTSCortex

RTSCortex is a hierarchical, real-time LLM agent runtime for StarCraft II. It keeps
environment control and PySC2 action execution in LLM-PySC2 while owning planning,
reflection, memory, reflexes, arbitration, and evaluation itself.

The deterministic mock adapter remains the fastest reproducible vertical slice. Live
smoke runs use an LLM-PySC2 worker isolated behind a versioned JSON protocol because its
Python and dependency requirements differ from the core runtime.

## Quick start

```bash
git submodule update --init
uv sync --group dev
uv run rtscortex doctor
uv run rtscortex run --config configs/experiments/mock_pvz.yaml
uv run rtscortex report ~/scratch/outputs/RTSCortex/<run-directory>
uv run pytest
```

Runtime outputs default to `~/scratch/outputs/RTSCortex`; environments and caches belong
under `~/fastscratch`. StarCraft II is not installed or downloaded by the core package.

On Linux, `rtscortex serve` creates a per-run Unix socket under the configured runtime
root. Pass `--tcp` to use loopback TCP instead.

The recommended compute-center live configurations are
`configs/experiments/live_simple64.yaml` and
`configs/experiments/live_simple64_qwen3_8b.yaml`. The companion
`configs/experiments/live_simple64_qwen3_8b_console.yaml` supplies the RGB and browser
settings for the read-only Live Console. Enable it explicitly with
`rtscortex run --console`; see
[the Live Console guide](docs/live-console.md) for SSH port forwarding, RGB-frame
semantics, and the non-persistence boundary. These configurations run a Protoss agent against a
VeryEasy Zerg built-in bot using the Macro build on the SC2 4.10-compatible `Simple64`
Ladder/Melee map. Both use `step_mul=1`, a fixed 0.25x simulation rate (5.6 game loops
per wall-clock second), and pause SC2 until the first plan is ready; later planning is
asynchronous. Replay saving is always disabled. These configurations are bounded live
smokes for validating the environment-to-action path, not complete win/loss matches.

The `2s3z` and original `pvz_task1_level1` configurations remain available as legacy
task-map scenarios. A live `run` validates the Python 3.9 worker, SC2 installation,
scenario-specific map path, model endpoint when configured, and required upstream
patches before it starts either process. Build-screen coordinates proposed by a plan are
resolved again against the current observation immediately before execution. See
[live runtime setup](docs/live-runtime.md) for the required layout, capability boundary,
and launch commands.

See [the architecture overview](docs/architecture/overview.md) for the data flow and
extension contracts.

Policy Comparison v0.2 provides a 48-state, six-stratum shadow benchmark for the current
Qwen planner and optional local HIMA Protoss specialists. It never dispatches candidate
actions and never downloads model weights. Start with the fully offline workflow:

```bash
uv run rtscortex policy-corpus verify benchmarks/policy/protoss_v0_2/manifest.yaml
uv run rtscortex policy-compare --config configs/policy/comparison_v0_2.offline.yaml
```

See [the Policy Comparison guide](docs/policy-comparison.md) for corpus provenance, HIMA
model gates, classification semantics, and output artifacts.
