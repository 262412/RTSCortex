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
```

After the live run, restore the clean pinned checkout by reversing exactly these reviewed
patches in reverse order:

```bash
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

The patches deliberately do not change camera calibration, team collection and ordering,
automatic economy, or other translator behavior. Observation mapping, runtime calls,
action routing, and execution feedback live in this bridge package rather than in the
upstream checkout.
