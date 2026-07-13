# Architecture overview

RTSCortex separates the StarCraft II environment worker from the agent runtime so each
process can use an appropriate Python and dependency set.

```text
LLM-PySC2 worker -- ObservationEnvelope --> Runtime
Runtime fast path: Reflex -----------------> Arbiter
Runtime slow path: Memory -> Reflection -> Planning -> Action -> Arbiter
Arbiter -- validated ActionBatch ----------> LLM-PySC2 translator
```

The canonical `SC2State` contains economy, production queue, own units, own structures,
and visible enemies. This is sufficient for the v0.1 runtime and deliberately matches the
semantic boundary needed by a future action-conditioned world model.

## Timing model

- Reflex policies run synchronously on every observation.
- Planning runs in the background in live mode and on a fixed game-loop cadence.
- The last valid plan remains active while a new plan is pending.
- Reflex commands can preempt only commands for the same actor scope.
- Every command has a priority, source, creation loop, and TTL.

## Safety boundary

Model responses are parsed into typed proposals. Only `ActionCommand` objects accepted by
the action validator can cross the environment boundary. Unknown actions, invalid argument
counts, invalid actor scopes, expired commands, and commands exceeding the action budget are
recorded and rejected.

## Research provenance

- [LLM-PySC2](https://github.com/NKAI-Decision-Team/LLM-PySC2) supplies the environment
  and structured action basis.
- [Orak](https://github.com/krafton-ai/Orak) informs the modular planning, reflection,
  memory, and evaluation flow; RTSCortex reimplements those concepts with typed contracts.
- [SwarmBrain](https://arxiv.org/abs/2401.17749) motivates the split between slow LLM
  strategy and fast deterministic reflexes.
- [StarWM](https://github.com/yxzzhang/StarWM) and
  [VLM-Play-StarCraft2](https://github.com/camel-ai/VLM-Play-StarCraft2) are future plugin
  targets and are not runtime dependencies in v0.1.
