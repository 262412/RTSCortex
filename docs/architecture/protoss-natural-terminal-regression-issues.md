# Protoss natural-terminal regression issue register

Status date: 2026-07-23

This register captures the open issues confirmed by the active Arbiter and active
CortexPlaybook Protoss natural-terminal regression on `Simple64`, Protoss versus
VeryEasy Zerg, seeds `0`, `1`, and `2`.

The three reference runs are:

| Seed | Run | Outcome | Steps |
|---|---|---|---:|
| 0 | `cortex-20260723T124530418998Z-1a0f8182` | draw | 39,599 |
| 1 | `cortex-20260723T140755995764Z-a0f2f53e` | defeat | 36,146 |
| 2 | `cortex-20260723T151245469240Z-6001d742` | defeat | 31,749 |

All three runs reached a natural SC2 terminal result without a Runtime crash.
Command lineage, terminal-report conservation, build verification, and production
verification remained intact. The issues below are therefore open gameplay-control
and strategic-control defects, not a claim that the Bridge is generally unstable.

## Open issues

### SCX-PT-001: Protoss worker production and gas economy do not sustain growth

- **Priority:** P0
- **Status:** implemented; natural-terminal acceptance pending
- **Components:** EconomyAgent, Protoss economy controller, Worker integration
- **Observed behavior:** all three runs reached a maximum of only 13 workers.
  The upstream automatic worker-training path emitted only one
  `Train_Probe_quick` action per run. Maximum stored vespene was only 48, 104,
  and 36 for seeds 0, 1, and 2.
- **Impact:** HIMA marks `TRAIN PROBE` as `managed_automatically`, but the
  automatic path does not provide the economy assumed by the Race Brain.
  Stargate requires 150 vespene, so no run built a Stargate or produced an
  anti-air Stargate unit.
- **Root-cause hypothesis:** RTSCortex delegates worker production to an
  opportunistic upstream automation path that has no persistent target, command
  lifecycle, or production-effect contract. Gas assignment also cannot overcome
  the resulting worker shortage.
- **Required correction:** make worker production and mineral/gas saturation an
  RTSCortex-owned EconomyAgent controller with tracked commands and effect
  verification. `managed_automatically` must mean managed by this controller,
  not silently delegated to an unverified side path.
- **Acceptance criteria:**
  - every worker-production attempt has command lineage and one terminal effect;
  - all three seeds reach at least 28 workers by 10 game minutes when a base is
    alive and worker production is not under an emergency hold;
  - completed Assimilators converge toward their configured worker saturation;
  - a committed Stargate plan can reserve and accumulate at least 150 vespene
    instead of having its gas consumed by lower-priority fallback actions.

### SCX-PT-002: Strict macro frontier blocks tech and permits structure oversaturation

- **Priority:** P0
- **Status:** implemented; natural-terminal acceptance pending
- **Components:** HIMA MacroPlan compiler, GoalProgress, Economy/Technology/Production roles
- **Observed behavior:** HIMA repeatedly proposed Stargate and air-unit
  transitions, but all three runs built zero Stargates. Seed 0 successfully
  issued eight Cybernetics Core builds, fourteen Shield Battery builds, and
  fourteen Pylon builds while still ending with one Nexus and two Gateways.
- **Impact:** model knowledge does not become a viable composition. A blocked
  expansion or gas-dependent frontier can hold back independent technology,
  while newly accepted plans can restart already-satisfied structure goals.
- **Root-cause hypothesis:** the current ordered frontier behaves too much like
  a strict sequence. It lacks a partial dependency graph, durable strategic
  commitments, and global structure-saturation/obsolete checks across plans.
- **Required correction:** compile macro proposals into partially ordered
  economy, supply, technology, production, and expansion commitments. A blocked
  Nexus must not block an independently legal Cybernetics Core or Stargate.
  Global state, rather than the current plan instance, must determine whether a
  tech or production structure is already sufficient.
- **Acceptance criteria:**
  - an invalid expansion candidate cannot block an otherwise viable tech path;
  - unique or saturation-limited structures are marked obsolete across plan
    revisions when their global target is already satisfied;
  - repeated HIMA plans do not produce redundant Cybernetics Core construction
    without an explicit validated capacity reason;
  - every committed tech transition records why it completed, remained
    deferred, was superseded, or was abandoned.

### SCX-PT-003: Expansion fallback suppresses a bad anchor but does not close the search

- **Priority:** P0
- **Status:** implemented; natural-terminal acceptance pending
- **Components:** expansion scouting controller, persistent world anchors, Builder
- **Observed behavior:** seed 0 tried one persistent expansion anchor, received
  `invalid_expansion_anchor`, and did not retry that same anchor. It also did not
  build a Nexus or emit expansion-candidate exhaustion. Seed 1 built one Nexus;
  seed 2 did not dispatch an expansion. Each run recorded only two expansion
  scout camera moves.
- **Impact:** permanent bad-anchor suppression works, but the expansion
  commitment can disappear without either success or a structured terminal
  explanation.
- **Root-cause hypothesis:** invalidation advances local anchor state, but no
  controller-owned state machine keeps the expansion objective alive while it
  scouts and evaluates the remaining resource clusters.
- **Required correction:** persist expansion commitments and iterate stable
  resource-cluster candidates until a Nexus is confirmed or the candidate set
  is explicitly exhausted.
- **Acceptance criteria:**
  - one expansion commitment terminates in exactly one of
    `nexus_effect_confirmed`, `expansion_candidates_exhausted`, or an explicit
    strategic cancellation;
  - an invalid anchor is never dispatched again in the same episode;
  - the next untried cluster is actively scouted without requiring it to be
    accidentally visible;
  - exhaustion records every evaluated anchor and rejection reason.

### SCX-PT-004: Combat target failures are retried without actor-local quarantine

- **Priority:** P0
- **Status:** implemented; natural-terminal acceptance pending
- **Components:** FocusFireAgent, CombatEffectVerifier, PlaybookCandidateGuard
- **Observed behavior:** seed 2 dispatched 92 `Attack_Unit` commands: 8
  succeeded, 55 ended with `combat_effect_not_observed`, 26 with
  `combat_target_lost`, 2 with `target_not_visible`, and 1 was cancelled at
  episode end. Target `0x100940002` was selected 29 times, including 28
  `combat_effect_not_observed` results.
- **Impact:** the effect verifier correctly rejects unproven attacks, but the
  tactical controller converts those failures into a retry storm instead of
  learning that the actor/target pairing is temporarily ineffective.
- **Root-cause hypothesis:** target reacquisition does not maintain an
  actor-local negative cache keyed by target, failure class, and observation
  time. A soft Playbook delta of `-0.5` is also too small to displace the only
  available focus-fire candidate.
- **Required correction:** add failure-counted target quarantine, cooldown,
  capability checks, and forced reacquisition. Playbook rules should refine
  this deterministic state machine, not compensate for its absence.
- **Acceptance criteria:**
  - the same actor/target pair cannot receive more than the configured bounded
    retries during one cooldown window;
  - `combat_effect_not_observed`, lost visibility, death, and attackability
    changes produce distinct target-state transitions;
  - when no current unit target is viable, Offense searches a structure or
    requests a composition capable of attacking the remaining enemy;
  - deterministic replay produces no unbounded repeated-target sequence.

### SCX-PT-005: Terminal collapse is misclassified as technology or production

- **Priority:** P1
- **Status:** implemented and covered by deterministic regression
- **Components:** Situation Intelligence v2, phase classifier
- **Observed behavior:** seed 1 ended with zero bases, zero army supply, and four
  Probes but was classified as `technology`. Seed 2 ended with zero bases, zero
  army supply, and six Probes but was classified as `production`.
- **Impact:** role agents and Playbook conditions receive a normal macro phase
  while the player is in a last-stand or elimination state.
- **Root-cause hypothesis:** phase selection is primarily driven by surviving
  production/tech structure types and does not give terminal military/economic
  collapse precedence.
- **Required correction:** introduce a deterministic crisis/last-stand override,
  or map the same evidence to a combat emergency, before normal early,
  technology, and production classification.
- **Acceptance criteria:**
  - zero town halls plus a visible living enemy can never be classified as a
    normal technology or production phase;
  - the phase fact records the collapse evidence and confidence;
  - Defense, Retreat, Economy, and Arbiter behavior for the crisis state has
    deterministic tests.

### SCX-PT-006: Threat level remains low through most active combat

- **Priority:** P0
- **Status:** implemented and reference-journal replay validated; live acceptance pending
- **Components:** Situation Intelligence v2, threat classifier, Live Console
- **Observed behavior:** no run emitted a `high` threat assessment. The
  distributions were:

  | Seed | none | low | high | critical |
  |---|---:|---:|---:|---:|
  | 0 | 225 | 1,449 | 0 | 2 |
  | 1 | 263 | 1,272 | 0 | 1 |
  | 2 | 370 | 988 | 0 | 6 |

  The rare `critical` values coincide with sparse `unit_under_attack` or
  `building_under_attack` alerts. The next observations fall back to `low` even
  while contact or base destruction continues.
- **Impact:** Defense, Retreat, Race Brain coordination, Arbiter emergency
  preemption, Playbook matching, post-game threat attribution, and the Live
  Console all receive a systematically understated battle state.
- **Confirmed root cause:** `_threat_level()` currently uses only the count of
  visible living enemies and a one-observation `under_attack` boolean. Without
  an alert, every non-empty enemy force is `low`, regardless of distance, army
  value, base proximity, damage, losses, attackability, or ongoing combat.
  There is no hysteresis or persistent threat episode.
- **Required correction:** replace the count-only rule with a stateful,
  evidence-based threat assessment. Inputs must include:
  - explicit attack alerts and recent damage/loss deltas;
  - nearest threat distance/ETA and proximity to bases, workers, and production;
  - enemy-versus-own combat value and air/ground capability mismatch;
  - army engagement/readiness and base/army collapse;
  - last positive threat evidence with hysteresis and a clear-resolution rule.
- **Acceptance criteria:**
  - an attack alert with a living enemy raises threat to at least `high` on the
    same observation;
  - an enemy force attacking a base, or a visible force overwhelming an empty
    army, is `critical`;
  - active contact with a comparable force is at least `high`, even after the
    one-frame alert disappears;
  - `high`/`critical` persists for a tested hysteresis window and clears only
    after the threat is dead, departed, or contradicted by newer evidence;
  - threat facts expose the score components and evidence in events, reports,
    and the Live Console;
  - replay fixtures from all three reference runs cover low, high, critical,
    escalation, persistence, and de-escalation.

## Implementation order

1. Fix SCX-PT-006 threat assessment and SCX-PT-001 worker/gas economy first;
   both feed every downstream strategic decision.
2. Replace the strict macro frontier and add global saturation checks
   (SCX-PT-002).
3. Close the expansion state machine (SCX-PT-003).
4. Add combat target quarantine and capability-aware reacquisition
   (SCX-PT-004).
5. Add the terminal-crisis phase override (SCX-PT-005).
6. Re-run the same three seeds from a frozen Playbook snapshot, then run a
   separate sequential shared-Playbook series to evaluate self-iteration.

## Implementation update: 2026-07-23

The code corrections in this register are now implemented. The issue statuses
remain explicit about natural-terminal acceptance because the fixes have not yet
been judged against a new three-seed full-match result.

### SCX-PT-001

- `Train_Probe` is now an RTSCortex action exposed by the Protoss melee profile.
- The Economy reflex controller continuously targets 22 workers per completed
  Nexus, capped at 80 workers.
- Probe commands use the normal command lifecycle and ProductionEffectVerifier;
  upstream opportunistic Probe training is disabled.
- Deterministic gas rebalance remains enabled, excludes the reserved Builder,
  and fills completed Assimilators to three workers while yielding to Runtime
  observations and effect verification.

### SCX-PT-002

- RaceProfile owns global structure saturation limits.
- Unique/saturation-limited structures are marked `obsolete` from global SC2
  state, including across revised HIMA plans.
- A deferred frontier can yield to a later legal action in a different
  economy/technology/production/defense domain without bypassing
  `runtime_frontier`, Candidate validation, or the Validator.
- Supply and resource fallbacks retain priority over generic independent work.

### SCX-PT-003

- An expansion proposal establishes a persistent Runtime commitment.
- `invalid_expansion_anchor`, `no_legal_placement`, and
  `target_not_created` are recoverable expansion failures: the macro step
  returns to `deferred` instead of freezing the entire plan.
- The Worker publishes `expansion_candidates_exhausted` through the next
  ObservationEnvelope.
- Each commitment emits `expansion_commitment_started`, zero or more
  `expansion_anchor_rejected` events, and exactly one
  `expansion_commitment_terminal` event.
- A commitment survives later macro-plan revisions by adding a synthetic
  town-hall frontier that still passes the normal legality checks.

### SCX-PT-004

- Focus-fire state is keyed by actor and target.
- Target failures have bounded retry counts and a 112-loop quarantine.
- Visibility loss, friendly/stale targets, and repeated unconfirmed effects
  have distinct transitions.
- Every transition is persisted as `tactical_target_state`; actors without a
  valid current-screen target fall back to search/navigation instead of
  repeating the quarantined pair.

### SCX-PT-005 and SCX-PT-006

- Situation Intelligence is stateful for each episode.
- Zero town halls plus a living enemy overrides normal technology/production
  classification with combat crisis.
- Threat scoring now incorporates attack alerts, recent damage and losses,
  base proximity, enemy/own combat value, empty-army collapse, missing anti-air,
  and town-hall loss.
- High/critical assessments persist through a 32-loop hysteresis window,
  including short observation gaps.
- Situation events, Markdown reports, and Live Console show the numeric score,
  evidence list, and hysteresis expiry.

Read-only reclassification of every stored protocol-v1.1 observation from the
three reference runs produced:

| Seed | none | low | high | critical | final phase | final threat |
|---|---:|---:|---:|---:|---|---|
| 0 | 222 | 953 | 487 | 14 | combat | low |
| 1 | 261 | 869 | 350 | 56 | combat | critical |
| 2 | 370 | 258 | 291 | 445 | combat | critical |

The replay did not mutate the historical journals. Unlike the original
classification, all three runs now contain sustained `high` evidence, and the
two destroyed-base terminal states end as combat/critical rather than normal
technology or production.

### Deterministic verification added

- Probe production is generated only below race-specific worker saturation.
- Builder-reserved workers cannot be selected for gas rebalance.
- A blocked expansion does not prevent an independently legal Stargate.
- An existing Cybernetics Core makes a revised redundant Core step obsolete.
- A failed expansion anchor is not retried and the next anchor is dispatched.
- Expansion-candidate exhaustion produces an explicit terminal state.
- Actor-local combat target quarantine prevents unbounded repeat selection.
- Threat escalation, unseen-target hysteresis, de-escalation, air-capability
  mismatch inputs, and terminal collapse are covered by unit tests.

The remaining acceptance action is a frozen-Playbook Protoss natural-terminal
regression on seeds `0`, `1`, and `2`, followed by comparison against the three
reference runs at the top of this file.

### Real Worker smoke

A 2,000-loop Simple64 seed-0 smoke completed through the real Python 3.9 Worker
and SC2 Base75689:

- run: `cortex-20260723T171015013124Z-85fadca6`;
- Runtime crash, unattributed primitive, and candidate-domain violation: zero;
- seven `Train_Probe` commands succeeded and were confirmed by exact Nexus
  production orders;
- workers increased to 18 within the bounded smoke;
- 11 of 12 meaningful commands succeeded; the final in-progress Gateway was
  correctly classified `unconfirmed` at truncation;
- one expansion commitment produced exactly one terminal state,
  `strategic_cancellation`, because the smoke deliberately ended at 2,000
  steps;
- all 74 threat assessments were `none`, consistent with no enemy contact in
  this short window.

This smoke validates the live execution path but does not replace the
natural-terminal three-seed acceptance thresholds.
