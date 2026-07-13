# LLM-PySC2 worker bridge

This package stays on Python 3.9 and communicates with the Python 3.11 RTSCortex runtime
through the versioned JSON API. It contains no planner or model client. Its pure bridge
modules and tests do not require PySC2 or a StarCraft II installation.

## PR3 worker boundary

The bridge has seven independently testable pieces:

1. `ObservationMapper` converts the live worker's JSON-safe snapshot
   into an `ObservationEnvelope`. Its state always contains economy, production queue,
   own units, own structures, and visible enemies.
2. `ActionRouter` converts an `ActionBatch` to upstream `Actions:` text. Actor identifiers
   use the canonical `agent/team` shape. Routes follow the current
   `team_unit_team_list` order and insert an explicit no-op for an empty team slot, because
   upstream consumes translated action lists positionally. Arguments declared as tags are
   rendered as hexadecimal values.
3. `ExecutionTracker` aggregates every PySC2 primitive produced by one translated command
   into one versioned `ExecutionReport`.
4. `BridgeCoordinator` is the transport seam. It maps one observation, calls `/v1/tick`
   once, prepares routes for enabled agents, and forwards completed execution reports.
5. `TimeStepExtractor` reads a PySC2 timestep without importing PySC2 and emits the
   JSON-safe five-part state plus per-team action schemas. Only `tag`, `screen`, and
   `minimap` arguments enter the v0.1 bridge.
6. `SharedDecisionBroker` is the multi-agent barrier. Every enabled upstream agent calls
   `get_text_o` once, but the final arrival performs the one shared runtime tick. Waiting
   agents then receive action text ordered by their own `team_unit_team_list`.
7. `RTSCortexLLMAgent` overrides the upstream `query` method only. It never calls the
   upstream model client and still sends routed text through upstream `get_func_a`.
   `RTSCortexMainAgent` retains upstream camera, grouping, economy, translator, and PySC2
   validation behavior while reporting observable primitive results and episode end.

The fixtures in `tests/fixtures/llm_pysc2` freeze both sides of the boundary without
requiring PySC2. Every available action carries explicit argument names and semantic types
(`tag`, `position`, and so on), and every team exposes `No_Operation`; this gives the core
validator enough information to reject malformed arguments and keeps its fallback action
routable to a canonical actor.

The worker command is intentionally inert on import. To launch a supported live scenario,
set `RTSCORTEX_RUN_ID` and `RTSCORTEX_EPISODE_ID`, then set either
`RTSCORTEX_RUNTIME_SOCKET` or `RTSCORTEX_RUNTIME_URL` and run
`rtscortex-llm-pysc2-worker`. The legacy `RTSCORTEX_SOCKET` name is also accepted.
`RTSCORTEX_SEED` is optional and defaults to zero; with the reviewed runner patch it is
passed into `SC2Env`, rather than only being recorded as metadata. The command launches
the scenario named by `RTSCORTEX_SCENARIO`, defaulting to `pvz_task1_level1`. The v0.1
worker supports `pvz_task1_level1` and the Linux-compatible `2s3z` smoke scenario.
Installing SC2 and accepting its license remain separate operator steps.

For `2s3z`, the worker uses upstream `ConfigSmac_2s3z` and preserves its positional team
order (`Zealot-1`, `Zealot-2`, `Stalker-1`). The bridge adds `No_Operation` to copied
per-config action lists when upstream omits it; it does not mutate the shared upstream
SMAC action constant. This keeps runtime fallback commands translatable and traceable to
an `ExecutionReport`. It also skips the unreachable large-map centering condition caused
by the small arena's camera boundary while leaving coordinate-range inference, grouping,
observation collection, and action execution in the upstream main loop.

## Upstream ownership

The upstream source is pinned at `third_party/LLM-PySC2`; do not edit the submodule
directly. Required main-loop changes are small reviewed patches in `patches/` and should
only be applied for live sessions, then reversed to restore the clean gitlink. Environment
semantics, camera control, team management, automatic economy, text-action translation,
and PySC2 action validation remain upstream-owned.

One upstream source patch changes the waiting-response branch to return one PySC2 no-op
per environment tick. This prevents its original busy-spin while the shared broker waits
for the runtime response. The second exposes the existing `SC2Env.random_seed` input on
the PySC2 runner so fixed-seed experiments control real game initialization.
