# Protoss natural-terminal regression issue register

Status date: 2026-07-24

This register contains only defects that remain open after the Slurm L40S
regression:

`protoss-fixed-natural-terminal-slurm-20260724T132144Z`

The run used the active Arbiter, an identical frozen Playbook baseline for each
seed, the HIMA Protoss a/b/c Ensemble, and `Simple64` Protoss versus VeryEasy
Zerg. There was no step limit.

| Seed | Run | Outcome | Steps | Meaningful success | Decisive observation |
|---|---|---|---:|---:|---|
| 0 | `cortex-20260724T132440635925Z-558d83b0` | defeat | 18,302 | 56/76, 73.7% | no expansion; only one attack produced damage |
| 1 | `cortex-20260724T140514321274Z-7c0d4a6f` | error | 197 | 39/42, 92.9% | out-of-range `select_point [2, -85]` terminated PySC2 |
| 2 | `cortex-20260724T141454658563Z-a693a4f7` | defeat | 26,938 | 96/128, 75.0% | 38 peak army supply, but 21 of 23 attacks failed |

The two valid natural-terminal episodes produced 152 successes from 204
meaningful commands (74.5%). Both were defeats.

## Open issues

### SCX-PT-014: orchestration can emit an out-of-range feature action and crash the episode

- **Priority:** P0
- **Status:** implementation complete; live three-seed verification pending
- **Components:** automatic worker management, Worker primitive safety,
  PySC2 boundary
- **Evidence:**
  - seed 1 terminated at step 197 with:

    ```text
    ValueError: Argument is out of range for 2/select_point
    got: [[SelectPointAct.select], [2, -85]]
    ```

  - the action was not a Runtime candidate and therefore did not increment
    `candidate_outside_pysc2_dispatches`;
  - immediately before termination the Worker was still producing valid
    Runtime decisions and execution reports;
  - upstream `main_agent_funcs.py` computes bounded `x, y` for `stop_worker`
    selection but calls `select_point` with the original `unit.x, unit.y`.
- **Impact:** a single optional automatic-worker orchestration primitive can
  bypass the Candidate/Validator boundary and terminate an otherwise healthy
  SC2 process.
- **Confirmed root cause:** Runtime semantic actions are validated before
  dispatch, but orchestration actions returned by upstream `main_agent_func*`
  do not pass through one final feature-action argument validator. The
  `stop_worker` path contains a direct variable-use defect: its clamped
  coordinates are discarded. PySC2 detects the invalid coordinate only inside
  `features.transform_action()`, where recovery is no longer possible.
- **Required correction:**
  1. fix the `stop_worker` selection patch to use the bounded coordinate;
  2. add one Worker-owned final primitive validator for every origin,
     including translator, orchestration, gas management, expansion scouting,
     camera, and automatic worker management;
  3. validate every point/rectangle/minimap coordinate and discrete enum
     against the active action specification;
  4. replace an invalid orchestration primitive with transport NoOp, clear its
     bounded chain, and record a structured recovery;
  5. keep an invalid translator primitive as a terminal command failure rather
     than silently clamping semantic intent.
- **Acceptance criteria:**
  - negative, NaN, and over-bound screen/minimap coordinates never reach
    `SC2Env.step()`;
  - the recorded seed 1 `[2, -85]` fixture produces recovery without process
    exit;
  - orchestration recovery cannot claim or complete a Runtime command;
  - seeds `[0,1,2]` have zero feature-action range exceptions.

### SCX-PT-015: targetability is not actor-specific, so ground units attack air units

- **Priority:** P0
- **Status:** implementation complete; live three-seed verification pending
- **Components:** Situation target facts, Tactical Agent, Candidate generation,
  Validator, CombatEffectVerifier
- **Evidence:**
  - seed 2 dispatched 23 `Attack_Unit` commands but only one damaged a target;
  - Adepts repeatedly targeted an Overseer and Mutalisks;
  - Zealots also targeted a Mutalisk and an Overseer;
  - seed 2 recorded 16 `combat_effect_not_observed` and five
    `combat_target_lost` failures;
  - seed 0 repeated the same defect against an Overseer.
- **Impact:** the Tactical layer can spend most combat windows issuing
  mechanically impossible attacks while a large army remains ineffective.
- **Confirmed root cause:** `living_targetable_enemies` currently means
  living, enemy, and feature-visible. It does not include the selected actor's
  weapon target domain. Candidate and Validator share the same enemy tag set,
  so neither distinguishes `ground`, `air`, and `both`. The final upstream
  alliance check prevents friendly fire but cannot reject a legal enemy that
  the actor cannot damage.
- **Required correction:**
  1. add RaceProfile combat capability metadata for each controlled unit type;
  2. derive `attackable_enemies_for_actor(actor)` from living enemy movement
     domain and actor weapon domain;
  3. use this exact set in Tactical selection, `argument_candidates`, dynamic
     schema, Candidate compilation, and Runtime validation;
  4. reject stale or domain-incompatible targets as
     `target_not_attackable_by_actor` before Bridge dispatch;
  5. keep CombatEffectVerifier as a final effect check, not the first place an
     impossible pairing is detected.
- **Acceptance criteria:**
  - Zealot and Adept candidates contain no flying unit tags;
  - Phoenix or other anti-air actors can receive air targets when enabled;
  - actor-domain validation is identical for Planner, Tactical, Reflex, and
    replay;
  - domain-incompatible attacks entering Bridge equal zero.

### SCX-PT-016: offense navigation never closes into enemy-structure destruction

- **Priority:** P0
- **Status:** implementation complete; live three-seed verification pending
- **Components:** Offense Agent, actor-local navigation, enemy memory,
  structure targeting
- **Evidence:**
  - neither valid long episode confirmed damage to an enemy structure;
  - seed 2 reached 38 peak army supply but still lost without converting that
    army into base damage;
  - successful movements repeatedly targeted the same nearby minimap points,
    especially `[39,47]` and `[41,46]`;
  - after unit targets disappeared, the agent alternated between waypoint
    movement and reacquisition instead of selecting an enemy building;
  - the post-game attributor recorded `advantage_not_converted` in seed 2.
- **Impact:** production can succeed and an army can cross the map, yet the
  system cannot complete the objective that ends a melee game.
- **Confirmed root cause:** the actor-local state machine tracks movement and
  current unit targets, but `last-known enemy structure` is used primarily as a
  navigation point. Arrival does not create a structure-search substate with a
  bounded camera sweep, target selection, attack confirmation, and waypoint
  retirement. Repeated waypoints remain eligible because successful one-tile
  displacement is treated as progress even when the group has not closed on
  the strategic target.
- **Required correction:**
  1. calculate navigation from the complete combat-group centroid;
  2. add explicit `travelling -> arrived -> searching -> attacking_structure
     -> cleared/failed` states per actor;
  3. retire a waypoint on arrival, repeated no-progress, timeout, or confirmed
     absence;
  4. preserve last-known living enemy structures separately from transient
     enemy units;
  5. after arrival, perform a bounded local search and emit an actor-compatible
     structure attack;
  6. use CombatEffectVerifier evidence to mark the structure damaged,
     destroyed, lost, or stale before selecting the next waypoint.
- **Acceptance criteria:**
  - repeated no-progress waypoint loops are zero;
  - an army with no living unit target searches for and attacks a reachable
    enemy structure;
  - each waypoint reaches one explicit terminal state;
  - a deterministic structure-search fixture produces a confirmed building
    attack lifecycle.

### SCX-PT-017: expansion commitment terminates before scouting can populate its anchor queue

- **Priority:** P0
- **Status:** implementation complete; live three-seed verification pending
- **Components:** ExpansionScoutController, persistent world anchors, Runtime
  expansion commitment, Nexus translator
- **Evidence:**
  - seeds 0 and 2 never exceeded one Nexus;
  - each commitment terminated near the opening with
    `expansion_candidates_exhausted` and `evaluated_anchors=[]`;
  - later in both games, two anchors were attempted outside the original
    commitment;
  - one anchor had no complete footprint with resource clearance and the next
    could not be found on the current screen;
  - the Worker reported seven/eight exhaustion alerts despite only three scout
    camera moves.
- **Impact:** the HIMA expansion objective is declared exhausted before active
  scouting has discovered and evaluated the map's candidate resource clusters.
- **Confirmed root cause:** the Runtime commitment and Worker scout have
  separate lifecycles. An empty initial persistent-anchor cache is interpreted
  as final exhaustion, while later discoveries can still be surfaced as
  stateless `Build_Nexus_Near` candidates. The episode exhaustion latch then
  prevents a coherent commitment from reopening. Anchor identity is also
  resource-unit based, while final Nexus placement requires a persistent world
  cluster plus a legal 5x5 center.
- **Required correction:**
  1. make the commitment own the scout waypoint queue, discovered resource
     clusters, rejected anchors, and placement candidates;
  2. distinguish `not_discovered_yet`, `search_in_progress`, and
     `all_candidates_exhausted`;
  3. persist cluster world bounds and ideal Nexus center, not only a currently
     visible resource tag;
  4. after invalidation, permanently suppress that candidate and immediately
     advance to the next unseen cluster;
  5. terminate only after all bounded scout waypoints have been visited and
     every discovered cluster has a terminal placement result.
- **Acceptance criteria:**
  - an empty opening cache cannot produce final exhaustion;
  - every expansion attempt is represented inside one commitment;
  - an invalid candidate is dispatched at most once;
  - the commitment ends with a confirmed Nexus or proof that all scout
    waypoints and clusters were exhausted;
  - at least one of seeds `[0,1,2]` builds a second Nexus on Simple64.

### SCX-PT-018: Builder selection drifts between observation and translator execution

- **Priority:** P1
- **Status:** implementation complete; live three-seed verification pending
- **Components:** Builder ownership, action availability, translator chain,
  build effect verification
- **Evidence:**
  - seed 0 emitted six consecutive Gateway failures at the same screen
    coordinate with `function Build_Gateway_screen is not available`;
  - the immediately preceding Runtime observations exposed
    `Build_Gateway_Screen` as available for the Builder;
  - seed 2 had one accepted Pylon whose expected target was not created and two
    accepted Assimilator actions with no observed build order;
  - seed 0 build pre-dispatch rejection was 10.5%; seed 2 was 7.4%.
- **Impact:** valid macro objectives are repeatedly lost to selection-state
  races, and the same failed signature can consume consecutive planning
  windows.
- **Confirmed root cause:** candidate availability is sampled from a fresh
  selected-Builder observation, but upstream automatic management and
  multi-agent orchestration can change feature selection before the final
  build primitive. The current bounded reselection is not an ownership lease:
  it does not prove that the exact Builder remains selected immediately before
  the final primitive. Failure deduplication is command-ID based, so a new plan
  can recreate the same action, coordinate, and failure.
- **Required correction:**
  1. create a Builder selection lease keyed by builder tag, semantic action,
     target, and observation loop;
  2. block optional selection-changing orchestration while the lease is active;
  3. require exact selected-tag evidence on the observation immediately before
     the build primitive;
  4. revalidate function availability and placement after that evidence;
  5. on failure, clear the chain and apply signature-scoped backoff until state
     or candidate changes;
  6. keep resource/order changes as diagnostics only.
- **Acceptance criteria:**
  - `observation available -> translator unavailable` mismatches equal zero;
  - the same failed build signature is not dispatched repeatedly without state
    change;
  - build pre-dispatch rejection is at most 5%;
  - build effect confirmed rate is at least 90%.

### SCX-PT-019: active Playbook scoring ignores recent terminal failure

- **Priority:** P1
- **Status:** implementation complete; live three-seed verification pending
- **Components:** PlaybookIntentGuard, PlaybookCandidateGuard, terminal
  feedback, rule specificity
- **Evidence:**
  - seed 0 recorded 73 non-zero rule applications and seed 2 recorded eight;
  - one soft role rule rewarded production candidates 62 times in seed 0,
    including `Build_Gateway_Screen` 48 times;
  - six identical Gateway translator failures occurred while that broad rule
    continued to add `+0.5`;
  - no Playbook application blocked a candidate;
  - seed 0 still ended with `production_imbalance`.
- **Impact:** CortexPlaybook now changes decisions, but it can reinforce a
  repeatedly failing action because its strategic preference is not combined
  with recent execution evidence.
- **Confirmed root cause:** soft rules are matched against Situation and
  candidate/role fields only. The Guard has no typed recent-terminal context
  keyed by action, actor, target, failure stage, and failure code. Role-only
  `prefer production` rules are consequently applied to every production
  candidate, even when the exact candidate just failed mechanically.
- **Required correction:**
  1. maintain a bounded recent-terminal feedback index by canonical action
     signature;
  2. apply deterministic cooldown/negative delta after retryable failure and a
     hard temporary suppression after non-retryable failure;
  3. release suppression only after relevant state, candidate, actor, or target
     changes;
  4. cap broad role-only bonuses so they cannot override a specific execution
     failure;
  5. record the failure evidence and final score components in lineage;
  6. do not promote episode-error or censored evidence directly to hard.
- **Acceptance criteria:**
  - a repeated unchanged translator failure is dispatched at most once during
    its cooldown;
  - a specific failure penalty dominates a broad role preference;
  - every non-zero Playbook delta identifies the applied rule or terminal
    feedback record;
  - false blocks remain below 5% and all temporary suppressions expire on
    relevant state change.

## Repair order

1. Add the universal primitive boundary and repair `stop_worker`
   (`SCX-PT-014`).
2. Add actor-specific target domains before generating any combat candidate
   (`SCX-PT-015`).
3. Close offense navigation into enemy-structure attack
   (`SCX-PT-016`).
4. Merge scouting and expansion commitment state (`SCX-PT-017`).
5. Add Builder selection lease and failure-signature backoff
   (`SCX-PT-018`).
6. Feed terminal outcomes back into active Playbook scoring
   (`SCX-PT-019`).
7. Run focused tests, all quality gates, then a new three-seed natural-terminal
   regression.

## Required engineering gates

```text
uv run pytest
uv run ruff check src tests integrations/llm_pysc2/src
uv run mypy
```

Worker Python 3.9 contract checks must additionally cover:

- out-of-range orchestration primitives;
- ground/air target compatibility;
- direct selection recovery and Builder selection leases;
- persistent expansion state;
- structure-search primitive attribution.

An item is not removed merely because unit tests pass. P0 live issues require a
new three-seed natural-terminal run to satisfy their acceptance criteria.
