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
uv run pytest
```

Runtime outputs default to `~/scratch/outputs/RTSCortex`; environments and caches belong
under `~/fastscratch`. StarCraft II is not installed or downloaded by the core package.

On Linux, `rtscortex serve` creates a per-run Unix socket under the configured runtime
root. Pass `--tcp` to use loopback TCP instead.

See [the architecture overview](docs/architecture/overview.md) for the data flow and
extension contracts.
