# RTSCortex Live Console

The Live Console is a read-only observability surface for one RTSCortex run. It shows
the RGB viewport that PySC2 provides to the agent, the current structured SC2 state,
planner/module progress, action validation, dispatch provenance, and effect-verification
outcomes. It cannot pause the game, inject an action, or mutate runtime state.

## Start a live console

Use the dedicated Simple64/Qwen example or add a `console` section to another live
configuration:

```bash
uv run rtscortex run \
  --config configs/experiments/live_simple64_qwen3_8b_console.yaml \
  --console \
  --console-port 8765
```

The console binds only to `127.0.0.1`. From a local computer, forward that loopback
port through SSH before opening the printed URL:

```bash
ssh -L 8765:127.0.0.1:8765 <compute-centre-host>
```

Then open <http://127.0.0.1:8765>. The port is checked before SC2 starts. If it is
already occupied, choose another value for both `--console-port` and the SSH tunnel.

Console settings are captured in the run's `config.yaml`:

```yaml
console:
  enabled: false
  port: 8765
  frame_fps: 2.0
  rgb_screen_size: 256
  rgb_minimap_size: 128
  jpeg_quality: 75
  stale_after_seconds: 2.0
  frontend_event_limit: 5000
```

`--console` enables the feature for that invocation and `--console-port` overrides the
configured port. With the console disabled, RTSCortex does not request RGB planes,
start a frame publisher, or bind a browser server.

## What the page shows

- **Game view:** the current `rgb_screen` and `rgb_minimap` observations. These are the
  agent-visible SC2 render planes, not a desktop capture of the Blizzard client and not
  a full HUD stream.
- **Planner activity:** module start time, elapsed time, provider/model, explicit
  Reflection and Planning results, context-compaction statistics, and token usage.
- **Goal progress:** deterministic completed/missing requirements, blockers, currently
  advancing actions, and the unique next action when one exists. This card is the state
  evidence supplied to both Reflection and Planning, not an LLM self-assessment.
- **Action trace:** proposals, validation and arbitration, dispatch, translator
  primitives, PySC2 acceptance, and ActionEffectVerifier evidence joined by command ID.
- **Timeline:** durable SQLite/JSONL events with filtering, pause/autofollow, reconnect
  backfill, and human-readable Chinese summaries for actions, states, stages, and failure
  codes.

The decision rail and event drawer use a presentation-only semantic layer. Protocol field
labels, SC2 actions, lifecycle states, execution stages, and known failure codes are shown
in Chinese while canonical action names, command IDs, actor scopes, model output prose,
coordinates, tags, and PySC2 function names remain unchanged for auditability. The original
event JSON is available under **技术详情：查看原始 JSON**, but stays collapsed by default.
This display transformation never rewrites the SQLite/JSONL journal or changes the Worker
control protocol.

The provider remains schema-based and non-streaming. The page therefore shows a live
elapsed timer while a module is running and displays its structured result when the
request completes; it does not expose hidden reasoning or stream partial JSON tokens.

## Storage and failure isolation

The Worker samples at the configured wall-clock frame rate and offers frames to a
single-slot queue. Encoding and Unix-socket upload happen in a background thread. A
new frame replaces an older queued frame, so a slow browser cannot build an unbounded
backlog or delay an SC2 step.

Only the latest screen and minimap JPEGs are retained in memory. They are never written
to the run directory, and replay saving remains disabled. SQLite and JSONL continue to
hold the durable decision events used for reports and reconnect backfill.

Closing the browser, losing the SSH tunnel, or overflowing a browser subscriber does
not stop the Runtime or Worker. The client reconnects with its last event ID and reads
any missed durable events from SQLite.

## Inspect a completed run

Serve a finished run with:

```bash
uv run rtscortex console ~/scratch/outputs/RTSCortex/<run-directory> --port 8765
```

Historical mode reconstructs the event timeline and decision panels. The game-view
panel explicitly reports that historical RGB is unavailable because frames are not
persisted.

## Security boundary

The loopback browser server exposes only `GET` and WebSocket observability endpoints.
The Worker continues to use the separate Unix-domain runtime API for tick, execution,
episode, and internal frame-ingest requests. The browser server never exposes those
control endpoints and never returns API keys or full system prompts.

## Frontend verification

The browser suite starts a disposable Python Console backed by the real SQLite event
store and WebSocket API. It checks the historical layout, command lifecycle drawer,
timeline filters, read-only route boundary, and event backfill after a forced socket
disconnect. Fixture data is created under the system temporary directory.

```bash
cd web/live-console
npm ci
npx playwright install chromium
npm run test:e2e
```

On a compute node with an existing Chromium installation, avoid another browser
download by providing its executable explicitly:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome npm run test:e2e
```
