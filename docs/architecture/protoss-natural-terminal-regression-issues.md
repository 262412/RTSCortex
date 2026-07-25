# Protoss natural-terminal regression issue register

Status date: 2026-07-24

This register contains only defects that remain open after the A100 regression:

`protoss-natural-terminal-py39fix-a100-20260724T171326Z`

The run used the active Arbiter, an identical frozen Playbook baseline for each
seed, and the HIMA Protoss a/b/c Ensemble on `Simple64` against VeryEasy Zerg.

| Seed | Run | Outcome | Steps | Meaningful success | Decisive observation |
|---|---|---|---:|---:|---|
| 0 | `cortex-20260724T171445861289Z-b77b61b6` | draw | 39,566 | 217/324, 67.0% | 86 peak army, one Nexus, map not closed |
| 1 | `cortex-20260724T183550105631Z-52be80c0` | draw | 39,566 | 192/262, 73.3% | 88 peak army, one Nexus, map not closed |
| 2 | `cortex-20260724T194708477833Z-6bc3aab6` | defeat | 17,982 | 49/52, 94.2% | critical threats, no Defense role intent |

The three episodes produced 458 successes from 638 meaningful commands
(71.8%). Build effects were confirmed 80/80 and production effects 111/111.
There were no candidate-domain, duplicate-dispatch, friendly-target, target
domain, or primitive-attribution violations.

The earlier out-of-range primitive defect and ground-unit-versus-air target
defect are therefore closed and have been removed from this register. Ordinary
Protoss build/producer provenance is also closed; the remaining Nexus failure is
tracked under the expansion lifecycle issue.

## Open issues

### SCX-PT-016: offense navigation never closes into enemy-structure destruction

- **Priority:** P0
- **Status:** partially implemented; awaiting natural-terminal live verification
- **Components:** Offense Agent, actor-local navigation, enemy memory,
  structure targeting
- **Evidence:**
  - seeds 0 and 1 reached 86/88 peak army supply but ended as draws;
  - 82/102 `Attack_Unit` commands confirmed damage, all against units;
  - neither episode produced an explicit enemy-structure attack lifecycle;
  - the army remained alive after the opponent's mobile force disappeared.
- **Impact:** a large army can win local fights but cannot destroy the final
  structures required to end a melee game.
- **Confirmed root cause:** current-screen unit targets and last-known
  navigation targets are implemented, but arrival at a strategic waypoint does
  not reliably transition into a bounded structure-search and structure-attack
  lifecycle. A movement command is also currently considered successful after
  observing any movement or a move order, so the Tactical state machine can
  retire or replace a waypoint without proving that the group arrived.
- **Required correction:**
  1. make movement completion mean actor-group arrival, not command start;
  2. retain `travelling -> arrived -> searching -> attacking_structure ->
     cleared/failed` per actor;
  3. retire a waypoint only on arrival, confirmed absence, or bounded timeout;
  4. preserve last-known enemy structures separately from transient units;
  5. require CombatEffectVerifier evidence to terminalize a structure target.
- **Acceptance criteria:**
  - one-tile displacement cannot confirm a remote `Move_Minimap`;
  - every offense waypoint reaches one explicit terminal state;
  - an army with no living unit target searches for and attacks a reachable
    enemy structure;
  - at least one deterministic fixture and one live episode confirm enemy
    structure damage.
- **Implemented on 2026-07-24:** movement is no longer terminalized by an
  order or partial displacement. The verifier snapshots the living actor tags,
  projects their centroid into minimap space, and waits for arrival. The
  existing actor-local offense/structure-search state machine will be
  re-evaluated against these corrected terminal reports in the paired live run.

### SCX-PT-020: natural-terminal configuration still inherits the map time limit

- **Priority:** P0
- **Status:** implemented; awaiting natural-terminal live verification
- **Components:** experiment configuration, PySC2 launch contract
- **Evidence:**
  - seeds 0 and 1 both ended at 39,566 agent steps with `draw`;
  - the natural-terminal config uses `game_steps_per_episode: null`;
  - Runtime omits the PySC2 flag for `null`;
  - the pinned `Simple64` map class defaults to `22 * 60 * 30 = 39,600`
    game loops.
- **Impact:** a time-limit draw is incorrectly interpreted as a natural SC2
  terminal, invalidating map-closure and paired Playbook conclusions.
- **Confirmed root cause:** `null` means “do not pass the override”, not
  “unlimited”. PySC2 uses `0` as the explicit no-limit sentinel.
- **Required correction:** set `game_steps_per_episode: 0` in every
  natural-terminal Protoss config and pin the resulting CLI flag in tests.
- **Acceptance criteria:**
  - launch command contains `--game_steps_per_episode 0`;
  - no natural-terminal result ends merely because game loop 39,600 was
    reached;
  - terminal outcomes come from SC2 victory/defeat or an explicit external
    failure.
- **Implemented on 2026-07-24:** both Protoss natural-terminal configurations
  now use `game_steps_per_episode: 0`; the config model accepts zero but rejects
  negative values, and the launch contract pins
  `--game_steps_per_episode 0`.

### SCX-PT-021: combat control-group selection and MoveVerifier use the wrong identity semantics

- **Priority:** P0
- **Status:** implemented; awaiting three-seed live verification
- **Components:** Protoss melee team configuration, Worker selection,
  movement effect verification
- **Evidence:**
  - the regression emitted 379 `cannot find unit` messages;
  - `Move_Minimap` succeeded only 185/339 times and 140 commands timed out;
  - the verifier currently succeeds when raw order 13 appears or when one
    representative unit moves at least one world unit;
  - the target is a minimap coordinate while the observed representative is a
    world coordinate;
  - Zealot/Stalker teams inherit upstream `select_type=group`, although
    RTSCortex does not own a durable create/update control-group lifecycle.
- **Impact:** selection can repeatedly recall stale groups, and “movement
  started” is reported as “destination reached”. Actor-local offense state is
  consequently driven by false terminal reports.
- **Confirmed root cause:** the Bridge binds a combat command to one historical
  head tag and the effect verifier observes that tag only. It neither snapshots
  the living tags in the routed actor nor projects their centroid into minimap
  space. The inherited control-group ID is mutable SC2 UI state outside
  RTSCortex provenance, so it cannot serve as an actor identity invariant.
- **Required correction:**
  1. use RTSCortex-owned exact-type/direct selection for controlled combat
     teams instead of unowned control-group recall;
  2. prune dead tags and deterministically rebind the team head before
     translation;
  3. snapshot all living actor tags and the world-to-minimap transform at
     dispatch;
  4. confirm movement only when the surviving group centroid is within the
     arrival radius of the requested minimap target;
  5. treat a move order and partial displacement as diagnostic progress only;
  6. derive an effective timeout from the initial target distance.
- **Acceptance criteria:**
  - stale control-group recalls equal zero;
  - a move order alone never produces `succeeded`;
  - a one-unit displacement toward a remote target remains pending;
  - group arrival within the configured radius succeeds;
  - `cannot find unit` and Move timeout rates both fall by at least 80% in the
    next three-seed regression.
- **Implemented on 2026-07-24:** controlled single-team combat actors no longer
  depend on unowned SC2 UI control groups. Effect preparation now carries all
  living routed tags plus the exact world-to-minimap transform; MoveVerifier
  uses surviving-group centroid arrival and a distance-derived timeout.

### SCX-PT-022: expansion commitment and background scouting have separate terminal states

- **Priority:** P0
- **Status:** implemented; awaiting three-seed live verification
- **Components:** ExpansionScoutController, persistent world anchors, Runtime
  expansion commitment, Nexus translator
- **Evidence:**
  - all three seeds ended with one Nexus;
  - 248 commitment terminal events were recorded, 246 with
    `evaluated_anchors=[]`;
  - only three expansion scout camera moves were recorded per seed;
  - a rejected anchor can reset Worker exhaustion without adding a new
    Runtime-owned commitment state;
  - seed 1 had one `Build_Nexus_Near` translator rejection.
- **Impact:** the Runtime can declare the strategic expansion attempt exhausted
  without proving that its bounded map search and all discovered clusters were
  evaluated.
- **Confirmed root cause:** Runtime owns commitment/evaluation events while the
  Worker independently owns waypoint, discovery, suppression, and exhaustion
  state. The only cross-process signal is a boolean exhaustion alert, which
  loses `not_discovered_yet`, `search_in_progress`, candidate identity, and
  sweep completion. Repeated exhaustion transitions therefore appear as many
  separate commitment terminals.
- **Required correction:**
  1. define one explicit scouting lifecycle:
     `not_discovered_yet -> search_in_progress -> candidate_available ->
     candidate_rejected/confirmed -> all_candidates_exhausted`;
  2. expose the lifecycle state and visited/total waypoint counts in every
     Worker observation;
  3. let one Runtime commitment persist across plans and anchor failures;
  4. keep rejected anchor tags permanently suppressed for the episode;
  5. terminalize only on confirmed townhall effect, strategic cancellation, or
     a completed sweep with every discovered candidate terminal.
- **Acceptance criteria:**
  - one HIMA expansion objective creates one commitment;
  - empty opening discovery cannot be final exhaustion;
  - each anchor is dispatched at most once;
  - `evaluated_anchors=[]` cannot accompany an exhausted commitment unless a
    complete zero-candidate sweep is recorded;
  - at least one of seeds `[0,1,2]` builds a second Nexus.
- **Implemented on 2026-07-24:** Worker observations now expose
  `not_discovered_yet`, `search_in_progress`, `candidate_available`, and
  `all_candidates_exhausted` together with visited/total waypoint counts.
  Runtime keeps a commitment through partial sweeps and refuses to accept the
  old boolean exhaustion alert when structured progress is incomplete.

### SCX-PT-023: DefenseAgent is a label router rather than a situation-response agent

- **Priority:** P0
- **Status:** implemented; awaiting three-seed live verification
- **Components:** Situation v2, RoleAgentCoordinator, DefenseAgent,
  Strategic Intent Arbiter
- **Evidence:**
  - seed 2 recorded 171 critical and 21 high-threat assessments;
  - seed 2 emitted zero Defense role intents;
  - all seven classes in `cortex/roles.py` currently inherit the same routing
    implementation;
  - Defense receives only an already-existing Reflex intent or a static-defense
    macro action.
- **Impact:** the Arbiter cannot select an emergency defense that no upstream
  producer proposed. High-quality threat detection therefore has no guaranteed
  path to unit response.
- **Confirmed root cause:** role ownership was implemented as post-hoc
  relabeling of Macro/Tactical/Reflex output. DefenseAgent has no state
  evaluation, no actor selection, no defensive target/region, and no
  hysteresis or commitment of its own.
- **Required correction:**
  1. make DefenseAgent independently inspect threat level, damage evidence,
     enemy proximity, and actor-compatible targets every observation tick;
  2. emit bounded emergency defense intents for available combat actors;
  3. prefer exact attackable threats; otherwise rally toward the threatened
     base region;
  4. add threat hysteresis and a short defense commitment to prevent flapping;
  5. retain routing only for legacy source intents and record independent
     Defense lineage distinctly.
- **Acceptance criteria:**
  - every high/critical fixture with an executable combat response emits at
    least one Defense intent;
  - incompatible actors/targets are never paired;
  - Defense wins actor conflicts against non-emergency Offense within 8 loops;
  - high/critical live intervals no longer have zero Defense role activity.
- **Implemented on 2026-07-24:** DefenseAgent now evaluates every current
  Situation, emits an emergency exact-target response or a threatened-base
  rally, and holds a short commitment. Immediate Reflex claims are passed into
  Defense so the two layers do not duplicate the same actor, while legacy
  creep/static-defense routing retains its non-emergency semantics.

### SCX-PT-024: a truncated HIMA cumulative action list discards an otherwise usable plan

- **Priority:** P1
- **Status:** implemented; awaiting live ensemble verification
- **Components:** HIMA generation contract, cumulative parser, Ensemble health
- **Evidence:**
  - seed 0 recorded 58 degraded coordination events;
  - Protoss-a emitted cumulative syntax such as `["Pylon": 3, ...]`;
  - generation reached `max_new_tokens=512` without EOS;
  - parser diagnostics were `output_truncated` and
    `action_section_missing`, even when complete counted entries existed before
    the cut.
- **Impact:** one ensemble member becomes unavailable during long games and the
  plan acceptance p95 grows to 207 loops.
- **Confirmed root cause:** complete cumulative lists are supported, and
  truncated ordinary string lists have prefix recovery, but truncated
  cumulative `"<token>": <count>` lists do not. The parser therefore cannot
  prove which prefix is complete and currently rejects the whole action
  section.
- **Required correction:**
  1. impose a bounded logical/effective action budget on cumulative output;
  2. recover only fully matched counted or bare entries before the truncated
     tail;
  3. reject an ambiguous/incomplete final entry;
  4. emit diagnostics that distinguish safe prefix recovery from unusable
     truncation;
  5. allow mapping only when the recovered prefix passes vocabulary and count
     conservation checks.
- **Acceptance criteria:**
  - a truncated cumulative list with at least one complete item yields a
    bounded proposal and `truncated_counted_prefix_recovered`;
  - the partial tail never becomes an action;
  - unknown tokens and invalid counts remain explicit parse errors;
  - recovered logical/effective counts conserve exactly;
  - a recoverable member is not marked degraded.
- **Implemented on 2026-07-24:** truncated-prefix recovery accepts only fully
  parsed bare or counted entries, drops an incomplete tail, and applies both
  logical-item and effective cumulative-count budgets. Mapping and ensemble
  health recognize `truncated_counted_prefix_recovered` as bounded recovery
  rather than member failure.

### SCX-PT-025: CortexPlaybook self-iteration has not been isolated from a frozen baseline

- **Priority:** P1
- **Status:** paired runner implemented; experiment pending
- **Components:** Playbook persistence, promotion sweep, paired evaluation
- **Evidence:**
  - non-zero Playbook deltas and terminal-feedback blocking occurred in the
    latest run;
  - the regression reset the same frozen database before every seed;
  - cross-game rule accumulation and its causal effect were therefore not
    tested.
- **Impact:** current evidence proves that active rules can affect a decision,
  but not that experience from one game improves a later matched game without
  increasing false blocks.
- **Confirmed root cause:** the experiment controls persistence for engineering
  reproducibility, while self-iteration requires a deliberately evolving
  database. No paired runner currently holds code/config/model/seeds constant
  while varying only Playbook persistence.
- **Required correction:**
  1. build paired `frozen` and `evolving` experiment arms from the same baseline;
  2. use identical seeds, model revisions, opponent, and natural-terminal
     settings;
  3. reset frozen before every game and carry evolving state across games;
  4. compare repeated-error signatures, false blocks, rule applications,
     promotions, score deltas, and outcomes;
  5. keep censored/error evidence ineligible for direct hard promotion.
- **Acceptance criteria:**
  - paired run metadata proves all non-Playbook variables are identical;
  - evolving state persists across seeds while frozen state does not;
  - classification and application counts conserve;
  - false-block rate remains below 5%;
  - repeated eligible error signatures decrease by at least 50%, or the result
    is explicitly reported as a failed Playbook-quality gate.
- **Implemented on 2026-07-24:** the paired runner alternates frozen/evolving
  arms for seeds `[0,1,2]`, restores frozen state before every episode, carries
  evolving state across episodes, snapshots both databases, and records source,
  config, baseline, run-directory, and before/after database hashes. The
  resulting six natural-terminal episodes are the remaining evidence gate.
- **Evidence run:** Slurm job `9917203` was submitted to
  `gpu-a100-lowbig` with two A100 GPUs. Its artifacts are rooted at
  `protoss-playbook-paired-natural-terminal-20260724T213200Z`.

## Repair order

1. Set the explicit unlimited PySC2 sentinel (`SCX-PT-020`).
2. Repair actor selection and true movement arrival (`SCX-PT-021`), which is
   also a prerequisite for offense closure (`SCX-PT-016`).
3. Merge expansion scouting and commitment state (`SCX-PT-022`).
4. Implement independent Defense response (`SCX-PT-023`).
5. Add bounded HIMA cumulative-prefix recovery (`SCX-PT-024`).
6. Run all engineering gates and a focused smoke.
7. Run the frozen/evolving paired Playbook experiment (`SCX-PT-025`).

## Required engineering gates

```text
uv run pytest
uv run ruff check src tests integrations/llm_pysc2/src
uv run mypy
```

Worker Python 3.9 contract checks must additionally cover:

- exact-type combat selection without unowned control-group state;
- group-centroid minimap arrival;
- expansion lifecycle transitions and commitment conservation;
- independent Defense intent generation;
- truncated cumulative HIMA prefix recovery.

An item is not removed merely because unit tests pass. P0 live issues require a
new three-seed natural-terminal run to satisfy their acceptance criteria.
