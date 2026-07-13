# RTSCortex

RTSCortex is a hierarchical, real-time LLM agent runtime for StarCraft II. It keeps
environment control and PySC2 action execution in LLM-PySC2 while owning planning,
reflection, memory, reflexes, arbitration, and evaluation itself.

The first implementation milestone is an offline vertical slice with a deterministic
mock SC2 adapter. A real LLM-PySC2 worker is intentionally isolated behind a versioned
JSON protocol because its Python and dependency requirements differ from the core
runtime.

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

The live configurations are `configs/experiments/live_pvz.yaml`, the Fake-provider
`configs/experiments/live_2s3z.yaml` smoke baseline, and
`configs/experiments/live_2s3z_qwen3_8b.yaml` for the local Qwen3-8B service. The 2s3z
configs run on the official Linux SC2 4.10 package and are the recommended
compute-center scenarios while the PvZ maps still require SC2 5.0.13. A live `run`
validates the Python 3.9 worker, SC2 installation, scenario-specific map path, model
endpoint when configured, and required upstream patches before it starts either
process. See [live runtime setup](docs/live-runtime.md) for the required layout and
launch commands.

See [the architecture overview](docs/architecture/overview.md) for the data flow and
extension contracts.
