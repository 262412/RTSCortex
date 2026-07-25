# Protoss natural-terminal regression issue register

Status date: 2026-07-25

This register contains only defects that remain open after the frozen/evolving
Playbook paired regression:

`protoss-playbook-paired-natural-terminal-20260724T213200Z`

The six runs used the same code, HIMA Protoss a/b/c Ensemble, active Strategic
Intent Arbiter, `Simple64`, Protoss versus VeryEasy Zerg, and explicit
`game_steps_per_episode: 0`. The frozen arm restored one baseline before every
seed; the evolving arm carried its database across seeds.

| Seed | Frozen | Evolving |
|---|---|---|
| 0 | defeat, 25,807 steps, 45/57 meaningful success | error, 14,904 steps, 43/71 |
| 1 | draw, 150,842 steps, 161/575 meaningful success | defeat, 19,360 steps, 39/67 |
| 2 | defeat, 21,636 steps, 45/54 meaningful success | draw, 74,861 steps, 75/337 |

The explicit no-limit configuration is now live-verified: neither draw ended at
the former 39,600-loop map default. HIMA reported no degraded member and no
truncated output in all six episodes. Those resolved issues have been removed
from this active register.

Production provenance, classification conservation, friendly-target safety,
duplicate-dispatch protection, and terminal-report exactly-once also remained
healthy. They are not repeated below.

## Open issues

### SCX-PT-016: offense navigation does not close the map

- **Priority:** P0
- **Status:** implemented; awaiting natural-terminal live verification
- **Components:** Offense Agent, actor-local navigation, enemy memory,
  CombatEffectVerifier
- **Evidence:**
  - no run won and every run finished with only one Nexus;
  - only frozen seed 1 confirmed enemy-structure damage, five times against an
    Extractor;
  - the two long games ended in SC2 `DrawAlert` despite peak armies of 33 and
    27 supply;
  - valid paired runs confirmed 348 frozen Move failures with zero successes,
    and 268 evolving failures with five successes.
- **Impact:** RTSCortex can produce an army and win local exchanges without
  navigating it to the remaining enemy structures, so melee victory is not
  reachable reliably.
- **Confirmed root cause:** the offense state machine consumes a command terminal
  as actor progress, but the Bridge and verifier disagree about actor identity.
  The feature action selects the currently visible exact unit type, while
  effect preparation snapshots every configured living team tag. A command can
  therefore be accepted for one selected subset while the verifier waits for a
  larger centroid that never received the order. Failed waypoints are then
  regenerated without a stable actor-local terminal/cooldown state.
- **Required correction:**
  1. capture the tags actually selected by the final feature observation and
     bind those exact tags to the Move command;
  2. require an expected raw move order on at least one bound tag before an
     accepted Move can remain in progress;
  3. compute arrival from surviving bound tags only;
  4. keep one actor-local waypoint lifecycle:
     `dispatched -> order_seen -> travelling -> arrived/failed/obsolete`;
  5. do not issue another waypoint for the same actor until the previous
     lifecycle is terminal;
  6. after arrival with no living unit target, search remembered enemy
     structures and require CombatEffectVerifier damage evidence.
- **Acceptance criteria:**
  - verifier actor tags equal the dispatched selected tags;
  - PySC2 acceptance without a move order cannot become success;
  - remote one-tile displacement remains pending;
  - the same actor cannot have overlapping navigation lifecycles;
  - Move true-arrival rate is at least 80% in deterministic contract tests and
    materially improves in the next live regression;
  - at least one live episode confirms enemy-structure damage after an
    actor-local arrival transition.
- **Implemented on 2026-07-25:** final Move preparation now binds the living
  feature-unit tags that were actually selected, not every configured team
  member. MoveVerifier requires raw order 13 within 16 loops, records the bound
  tags, and reports `move_order_not_observed` separately from arrival timeout.
  Move terminal feedback now advances or obsoletes the actor-local
  offense/retreat waypoint before another intent can be emitted.

### SCX-PT-021: stale combat team heads can starve Runtime observations

- **Priority:** P0
- **Status:** implemented; awaiting three-seed live verification
- **Components:** Worker team membership, MainAgent camera/selection chain,
  observation-gap watchdog
- **Evidence:**
  - evolving seed 0 terminated with
    `observation_gap_watchdog_timeout: no Runtime decision for 449 game loops`;
  - the run repeatedly attempted `CombatGroup3` with a stale `VoidRay-1` tag and
    logged 132 `cannot find unit` messages for that group;
  - all runs still produced `cannot find unit`; the valid evolving rate was
    77 occurrences over 94,221 loops;
  - after the soft watchdog fired, upstream continued to return
    `Reach MAX_LLM_DECISION_FREQUENCY! return no_op()`.
- **Impact:** one stale team head can monopolize camera/selection orchestration,
  prevent new Runtime decisions, terminate a long experiment, and feed
  infrastructure failures into downstream learning.
- **Confirmed root cause:** recovery currently clears optional func4 work and
  an observed pending orchestration primitive, but it does not atomically reset
  all upstream agent queues and the stale team head. More importantly, the
  upstream frequency guard runs after optional functions and before the main
  Runtime query. It does not exempt `_rtscortex_force_runtime_decision`, so the
  watchdog can request recovery yet still be returned as transport NoOp until
  the hard limit fires.
- **Required correction:**
  1. make the upstream frequency guard bypassable only by the explicit
     RTSCortex force-decision flag;
  2. at soft-watchdog activation atomically abort the active orchestration
     chain, clear queued team actions, clear the current team head, and
     terminalize its command once;
  3. remove confirmed-dead tags from team membership and permanently quarantine
     them for the episode;
  4. rebind a living replacement deterministically before another translation;
  5. make the same observation reach Runtime in the watchdog activation tick;
  6. reset watchdog recovery state only after observing a newer Runtime decision.
- **Acceptance criteria:**
  - a forced Runtime decision bypasses frequency throttling;
  - a synthetic stale camera chain produces one terminal abort and a Runtime
    observation before the hard limit;
  - dead tags cannot return to team queues;
  - no command receives two terminal reports during recovery;
  - three live seeds have zero watchdog hard-limit termination and no repeated
    `cannot find unit` loop for one tag.
- **Implemented on 2026-07-25:** reviewed upstream patch 0022 exempts only an
  explicit forced Runtime decision from the frequency throttle. Every Worker
  observation now prunes confirmed-dead combat tags, quarantines them, and
  deterministically rebinds a living team head. Watchdog recovery clears the
  current actor identity together with its camera/selection chain.

### SCX-PT-022: expansion search exhaustion is not generation-stable

- **Priority:** P0
- **Status:** implemented; awaiting three-seed live verification
- **Components:** ExpansionScoutController, persistent world anchors, Runtime
  expansion commitment, Nexus translator
- **Evidence:**
  - all six runs had a maximum of one Nexus;
  - the runs emitted 1,791 expansion terminals with empty
    `evaluated_anchors`;
  - each run issued only three scout camera moves;
  - the Worker alternated `candidate_available` and
    `all_candidates_exhausted`, allowing Runtime to create a fresh commitment
    for the same already exhausted search.
- **Impact:** HIMA can repeatedly request expansion without ever completing one
  bounded search, selecting a new anchor, or building a second Nexus.
- **Confirmed root cause:** exhaustion is stored as a mutable boolean rather
  than a generation-owned terminal. `candidate_available` unconditionally
  clears Runtime exhaustion, even if the candidate belongs to the same sweep.
  After commitment termination, the next macro proposal creates a new
  commitment against the same three visited waypoints. Worker and Runtime do
  not share a stable search generation ID or the generation's discovered,
  rejected, and evaluated anchor sets.
- **Required correction:**
  1. give every explicit expansion search a monotonic `generation_id`;
  2. include generation, visited/total waypoints, available anchors and rejected
     anchors in every structured observation;
  3. make one Runtime commitment own one generation and survive candidate
     invalidation;
  4. terminalize exhaustion once per generation and latch it until a deliberate
     new strategic search generation is requested;
  5. never let `candidate_available` from the same generation clear a terminal;
  6. expand the waypoint sweep beyond the current three points and record full
     zero-candidate coverage explicitly.
- **Acceptance criteria:**
  - one generation creates at most one commitment and one terminal;
  - the same anchor is dispatched at most once per episode;
  - an exhausted generation cannot restart from an unchanged Worker state;
  - an empty evaluated-anchor list is legal only after a recorded complete
    zero-candidate sweep;
  - at least one of seeds `[0,1,2]` builds a second Nexus or reports a single
    explicit full-map exhaustion terminal.
- **Implemented on 2026-07-25:** ExpansionScoutController now emits a stable
  generation ID plus available/rejected anchors, keeps an exhausted generation
  latched, and fills resource-cluster waypoints with eight deterministic
  map-spanning points. Runtime binds commitments and terminals to the generation
  and refuses to recreate a commitment when a candidate from the same exhausted
  generation reappears.

### SCX-PT-023: DefenseAgent emits an emergency intent storm

- **Priority:** P0
- **Status:** implemented; awaiting three-seed live verification
- **Components:** Situation v2, DefenseAgent, Strategic Intent Arbiter,
  Playbook Intent Guard
- **Evidence:**
  - Defense now responds, but evolving seed 2 emitted 2,054 Defense intents;
  - only 13 Defense commands succeeded while 188 failed;
  - 181 of those failures were Move effect timeouts;
  - broad active `prefer defense` rules applied thousands of times and added
    score to repeated unresolved Defense intents.
- **Impact:** a valid threat signal can cause Defense to occupy the Arbiter and
  repeatedly dispatch the same ineffective movement instead of maintaining one
  bounded response or allowing macro/offense progress.
- **Confirmed root cause:** DefenseAgent has one global
  `_committed_until_game_loop`, but emits a new step-specific intent on every
  high/critical observation. The commitment is only used when threat drops; it
  is not an actor-local active intent. The Agent receives no terminal feedback,
  has no target signature, no arrival/obsolete state, and no post-failure
  cooldown. Soft Playbook preference then amplifies the same unresolved action.
- **Required correction:**
  1. keep actor-local Defense state keyed by actor and target signature;
  2. suppress duplicate intents while the actor's response is active;
  3. add high/critical entry hysteresis, medium-threat hold, arrival/target-loss
     obsolescence and failure cooldown;
  4. allow exact attackable threats to replace a rally, but not another
     identical rally;
  5. prevent broad Playbook `prefer defense` from stacking on an unresolved
     Defense action.
- **Acceptance criteria:**
  - one actor/target signature emits at most one intent within its commitment;
  - Defense still preempts non-emergency Offense within 8 loops;
  - target disappearance makes an Attack intent obsolete;
  - failed Defense Move observes a cooldown before retry;
  - Defense intent volume is bounded by threat transitions and command
    terminals rather than observation count.
- **Implemented on 2026-07-25:** DefenseAgent now owns per-actor response
  signatures with a 112-loop commitment and failure cooldown. Only a precise
  Attack may replace an unresolved rally. Command lineage feeds terminal
  ExecutionReports back to Defense, so success clears the state and failure
  prevents immediate re-emission.

### SCX-PT-025: Playbook iteration works mechanically but failed its quality gate

- **Priority:** P1
- **Status:** partially implemented; quality experiment remains open
- **Components:** Playbook review, promotion eligibility, paired evaluation
- **Evidence:**
  - the evolving database grew from 486 to 911 cases and from 36 to 62 rules;
  - active soft rules grew from 6 to 16 and false-block rate was 0/6,775;
  - evolving seed 2 recorded 4,707 non-zero deltas across 7,708 applications;
  - on valid paired seeds, exact strategic consequences were 7 frozen versus 7
    evolving, and meaningful success fell from 32.75% to 28.22%;
  - evolving seed 0 ended in an infrastructure error but still wrote 51 cases,
    which contributed evidence to later soft promotion.
- **Impact:** the system can learn and influence decisions while learning from
  invalid infrastructure outcomes or amplifying broad preferences that do not
  reduce repeated errors.
- **Confirmed root cause:** error episodes are excluded from strategic
  consequence attribution but ordinary execution cases from those episodes are
  still consolidated into executable rules. Soft promotion counts all source
  runs/seeds and does not subtract censored/error sources. The paired experiment
  also has only one nondeterministic episode per seed, so identical starting
  hashes can diverge before Playbook effects and invalidate single-run causal
  attribution.
- **Required correction:**
  1. retain error-episode cases for diagnostics but mark them
     `promotion_eligible=false` and censored;
  2. prevent error/censored runs and seeds from satisfying soft as well as hard
     promotion support;
  3. record the exclusion reason in rule evidence and promotion reports;
  4. suppress or suspend broad soft preferences that repeatedly reinforce a
     terminally failing action family;
  5. rerun from a fresh baseline only after navigation and watchdog defects are
     fixed, with repeated trials or deterministic journal replay.
- **Acceptance criteria:**
  - an error-only case can never create or promote an executable rule;
  - mixed rules count only uncensored run/seed support for promotion;
  - diagnostic cases remain queryable and retain their failure lineage;
  - paired evaluation reports invalid pairs separately from quality metrics;
  - repeated eligible error signatures decrease by at least 50%, or the quality
    gate is explicitly failed without claiming improvement.
- **Implemented on 2026-07-25:** error-episode cases remain queryable but carry
  `promotion_eligible=false`, `censored=true`, an exclusion reason, and the
  episode failure. They no longer create or merge executable rules, contribute
  contradictions, or count as uncensored soft-promotion run/seed support. A
  fresh paired quality experiment is still required after the P0 live gates.

## Repair order

1. Repair forced watchdog preemption and stale combat-team identity
   (`SCX-PT-021`).
2. Bind Move verification to the actually selected actor tags and stabilize
   actor-local navigation (`SCX-PT-016`).
3. Introduce generation-owned expansion search and commitment state
   (`SCX-PT-022`).
4. Add actor-local Defense deduplication, hysteresis and cooldown
   (`SCX-PT-023`).
5. Exclude infrastructure-error evidence from executable Playbook promotion
   (`SCX-PT-025`).
6. Run focused Worker Python 3.9 contracts and core tests before another live
   natural-terminal regression.

## Required engineering gates

```text
uv run pytest
uv run ruff check src tests integrations/llm_pysc2/src
uv run mypy
```

Worker Python 3.9 checks must additionally cover:

- forced Runtime frequency bypass and watchdog same-tick recovery;
- stale-tag quarantine and deterministic team-head rebind;
- selected-tag Move provenance, raw move order and true arrival;
- expansion generation/commitment conservation;
- actor-local Defense intent deduplication and cooldown;
- error-episode Playbook promotion exclusion.

An item is not removed merely because unit tests pass. P0 live issues require a
new three-seed natural-terminal run to satisfy their acceptance criteria.
