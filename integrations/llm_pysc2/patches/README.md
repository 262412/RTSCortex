# Upstream hook patches

The submodule remains pinned and read-only between live sessions. Apply patches from
inside the pinned checkout before a live run:

```bash
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0003-fix-build-feature-plane-coordinate-order.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0003-fix-build-feature-plane-coordinate-order.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0004-expose-structured-translation-results.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0004-expose-structured-translation-results.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0005-fix-near-base-placement.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0005-fix-near-base-placement.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0006-report-pretranslation-action-aborts.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0006-report-pretranslation-action-aborts.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0007-preserve-transient-team-units.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0007-preserve-transient-team-units.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0008-enforce-nexus-resource-clearance.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0008-enforce-nexus-resource-clearance.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0009-use-exact-nexus-screen-scale.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0009-use-exact-nexus-screen-scale.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0010-report-max-frame-truncation.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0010-report-max-frame-truncation.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0011-allocate-log-directories-atomically.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0011-allocate-log-directories-atomically.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0012-bind-gas-rebalance-to-worker-management.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0012-bind-gas-rebalance-to-worker-management.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0013-preserve-reserved-builder-worker.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0013-preserve-reserved-builder-worker.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0014-refresh-worker-workplaces.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0014-refresh-worker-workplaces.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0015-observation-gap-watchdog.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0015-observation-gap-watchdog.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0016-accept-visible-team-unit.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0016-accept-visible-team-unit.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0017-return-camera-settlement-noop.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0017-return-camera-settlement-noop.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0018-use-exact-single-unit-selection.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0018-use-exact-single-unit-selection.patch
git -C third_party/LLM-PySC2 apply --check \
  ../../integrations/llm_pysc2/patches/0019-bypass-actor-selection-for-transport-noop.patch
git -C third_party/LLM-PySC2 apply \
  ../../integrations/llm_pysc2/patches/0019-bypass-actor-selection-for-transport-noop.patch
```

After the live run, restore the clean pinned checkout by reversing exactly these reviewed
patches in reverse order:

```bash
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0019-bypass-actor-selection-for-transport-noop.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0019-bypass-actor-selection-for-transport-noop.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0018-use-exact-single-unit-selection.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0018-use-exact-single-unit-selection.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0017-return-camera-settlement-noop.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0017-return-camera-settlement-noop.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0016-accept-visible-team-unit.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0016-accept-visible-team-unit.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0015-observation-gap-watchdog.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0015-observation-gap-watchdog.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0014-refresh-worker-workplaces.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0014-refresh-worker-workplaces.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0013-preserve-reserved-builder-worker.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0013-preserve-reserved-builder-worker.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0012-bind-gas-rebalance-to-worker-management.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0012-bind-gas-rebalance-to-worker-management.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0011-allocate-log-directories-atomically.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0011-allocate-log-directories-atomically.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0010-report-max-frame-truncation.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0010-report-max-frame-truncation.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0009-use-exact-nexus-screen-scale.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0009-use-exact-nexus-screen-scale.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0008-enforce-nexus-resource-clearance.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0008-enforce-nexus-resource-clearance.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0007-preserve-transient-team-units.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0007-preserve-transient-team-units.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0006-report-pretranslation-action-aborts.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0006-report-pretranslation-action-aborts.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0005-fix-near-base-placement.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0005-fix-near-base-placement.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0004-expose-structured-translation-results.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0004-expose-structured-translation-results.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0003-fix-build-feature-plane-coordinate-order.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0003-fix-build-feature-plane-coordinate-order.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0002-pass-random-seed-to-sc2env.patch
git -C third_party/LLM-PySC2 apply --reverse --check \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
git -C third_party/LLM-PySC2 apply --reverse \
  ../../integrations/llm_pysc2/patches/0001-return-noop-while-awaiting-runtime.patch
```

Do not configure Git to ignore dirty submodules: that would also hide accidental upstream
edits or gitlink drift.

`0001-return-noop-while-awaiting-runtime.patch` changes one branch in `MainAgent.step`.
The upstream implementation currently spins inside its bounded `while` loop while an
agent thread waits for a response. RTSCortex planning is asynchronous, so that branch
returns a PySC2 no-op for the current environment tick instead. The next environment tick
re-enters the normal upstream loop and observes the completed response when available.

`0002-pass-random-seed-to-sc2env.patch` adds a `--random_seed` runner flag and passes it
to `SC2Env`. This makes the seed recorded in experiment provenance control the actual
game initialization as well.

`0003-fix-build-feature-plane-coordinate-order.patch` corrects build validation to index
PySC2 feature planes in their row-major `[y][x]` order. It changes only the power, creep,
buildable, pathable, and player-relative reads in `get_arg_screen_build`.

`0004-expose-structured-translation-results.patch` records every translator attempt with
the requested and emitted function IDs, ordinal, chain length, acceptance result, and raw
failure reason. The final attempt also records the actual PySC2 arguments selected by the
translator. Multi-argument actions accumulate validity across every argument, so a valid later
argument cannot overwrite an earlier rejection. The bridge uses this provenance instead of
guessing command completion or resolved placement from function IDs.

`0005-fix-near-base-placement.patch` makes Near-build helpers preserve the requested
neutral resource anchor, preserve equal-distance resources, reject undersized or occupied
expansion clusters, and validate a complete empty footprint with row-major feature-plane
indexing. A cluster near an existing Protoss, Terran, or Zerg townhall of either player is
occupied. Nexus chains include one translator-owned no-op after moving the camera so SC2's
feature-unit visibility catches up before screen placement is resolved. A missing, stale,
off-screen, or out-of-bounds anchor is rejected instead of being replaced by another resource
tag.

`0006-report-pretranslation-action-aborts.patch` keeps team membership tied to raw-unit
existence instead of transient feature-screen visibility. Observation collection waits one
transport tick when a raw unit has not appeared on the new feature screen instead of deleting
the live unit from its team. If an action still cannot reach the translator because its actor is
gone or not visible, MainAgent emits a structured abort marker. The bridge consumes that marker
as an exact `pre_dispatch` failure, releases the active actor, and prevents a silently dropped
command from blocking every later command for that team. Patch 0007 supersedes 0006's immediate
raw-absence removal rule with confirmed disappearance.

`0007-preserve-transient-team-units.patch` makes the upstream 40-step disappearance counter the
single death authority. A Probe inside an Assimilator or a unit inside transport can disappear
from both raw and feature observations without losing its team or terminating an action. Only a
confirmed disappearance removes membership and produces `actor_not_available`. The counter is
promoted to confirmed-death state on every environment observation, including while MainAgent
holds its action-loop lock, so a dead team head cannot block the lock that would confirm it.
Upstream transport-level `No_Operation` actions never emit semantic abort markers because they
do not own an RTSCortex command.

`0008-enforce-nexus-resource-clearance.patch` replaces centroid-first Nexus placement with a
deterministic constrained search. Every candidate must preserve the established mineral
6-to-9-world-unit and geyser 7-to-10-world-unit distance bands, pass the complete 5x5 footprint,
and remain clear of currently visible townhalls. Stable scoring favors the ideal resource-ring
distance before centroid proximity.

`0009-use-exact-nexus-screen-scale.patch` separates the floating-point screen-to-world scale
from the integer sampling stride. Resource-ring and townhall distances therefore use the exact
`size_screen / 24` conversion, while footprint bounds cover the full 5x5 Nexus area. The exact
anchor and every footprint pixel must also be currently visible before placement is emitted.

`0010-report-max-frame-truncation.patch` invokes an optional agent lifecycle hook before
PySC2 returns at `max_agent_steps`. The RTSCortex bridge uses the hook to post one explicit
`truncated` episode result and finalize pending commands instead of relying on the supervisor
to guess why a clean worker process exited.

`0011-allocate-log-directories-atomically.patch` replaces the process-local logger counter
condition with an atomic `mkdir` claim. Concurrent live workers that start within the same
second now select distinct upstream log directories instead of one worker spinning forever.

`0012-bind-gas-rebalance-to-worker-management.patch` makes mineral-to-gas worker rebalancing
obey `ENABLE_AUTO_WORKER_MANAGE` instead of the unrelated worker-training flag. The Bridge
selects the exact nearest mineral Probe with stable tag tie-breaking before the upstream
stop-and-reassign sequence runs.

`0013-preserve-reserved-builder-worker.patch` prevents upstream idle-worker selection from
reassigning the dedicated Builder Probe to minerals or gas. The Bridge publishes reserved
Builder tags before worker management; if PySC2 selects one as idle, upstream gives it a stable
HoldPosition order and continues the ordinary worker rebalance on a later tick. The melee profile
keeps worker management enabled so the deterministic gas rebalance can run; if the Builder is
already present in a gas slot, the Bridge evicts that exact tag before filling the slot with an
ordinary worker.

`0014-refresh-worker-workplaces.patch` rebuilds each Nexus resource list from the current raw
observation so depleted mineral fields and destroyed gas buildings cannot remain valid worker
targets. Before assignment, it revalidates the selected Nexus and workplace, prefers the newest
observation-bound entry, and returns a transport no-op when no current resource target remains
instead of indexing an empty list.

`0015-observation-gap-watchdog.patch` lets the Worker skip optional upstream team gathering
when the gap since the latest Runtime decision exceeds the configured game-loop threshold.

`0016-accept-visible-team-unit.patch` stops camera centering from interrupting the existing
point-to-rectangle selection fallback when a team unit is already visible. RTSCortex enables
this behavior because its Runtime observation is global; selection remains required later for
actual feature actions.

`0017-return-camera-settlement-noop.patch` yields the current SC2 step after a production
camera move. The producer visibility timeout therefore counts distinct PySC2 observations
instead of repeated translator calls within one upstream waiting loop.

`0018-use-exact-single-unit-selection.patch` makes RTSCortex single-unit teams use exact point
selection instead of the upstream exponentially expanding rectangle fallback. This prevents a
Builder or exact production actor from remaining unselected until the rectangle radius overflows,
while leaving the original fallback available to non-RTSCortex users.

`0019-bypass-actor-selection-for-transport-noop.patch` executes RTSCortex transport-level
`No_Operation` directly. A control no-op has no actor semantics, so it must not move the camera
or select a Builder, producer, or combat unit merely to idle that team. This prevents idle teams
from starving Runtime observations through repeated feature-layer selection attempts.

CI applies all nineteen patches in order under Python 3.9, compiles and imports both projects, and
runs `integrations/llm_pysc2/tests/python39_contract_smoke.py`. The smoke locks the v1.1
candidate mapping, multi-argument translator rejection, Nexus camera-settlement primitive,
exact Nexus anchor, floating-point resource clearance, visible complete-footprint behavior,
gas clearance, raw-unit team persistence,
and explicit pre-translation abort markers, including transient disappearance recovery. CI then
reverses the patches in reverse order and requires a clean submodule checkout.

The patches deliberately do not change camera calibration, initial team assignment, or unrelated
translator behavior. Patch 0007 preserves insertion order while removing
duplicate team tags and delays member removal until death is confirmed. Observation mapping,
runtime calls, action routing, and execution feedback live in this bridge package rather than in
the upstream checkout.
