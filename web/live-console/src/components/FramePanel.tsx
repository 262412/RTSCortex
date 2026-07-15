import { Expand, ImageOff, Map, Monitor } from "lucide-react";
import { useEffect, useState } from "react";

import { frameUrl } from "../api";
import type { ConnectionStatus, FrameMetadata } from "../types";

interface FrameViewProps {
  kind: "screen" | "minimap";
  frame?: FrameMetadata;
  connection: ConnectionStatus;
  staleAfterSeconds: number;
  historical: boolean;
}

function FrameView({ kind, frame, connection, staleAfterSeconds, historical }: FrameViewProps) {
  const [failedSequence, setFailedSequence] = useState<number | null>(null);
  const [ageSeconds, setAgeSeconds] = useState<number | null>(null);

  useEffect(() => {
    const capturedAt = frame?.captured_at ? Date.parse(frame.captured_at) : Number.NaN;
    const timer = window.setInterval(() => {
      setAgeSeconds(Number.isFinite(capturedAt) ? Math.max(0, (Date.now() - capturedAt) / 1_000) : null);
    }, 1_000);
    return () => window.clearInterval(timer);
  }, [frame?.captured_at]);

  const imageFailed = frame !== undefined && failedSequence === frame.frame_sequence;
  const age = ageSeconds;
  const stale = age !== null && age > staleAfterSeconds;
  const indicator = historical
    ? "Unavailable"
    : connection === "disconnected" || connection === "reconnecting"
      ? "Disconnected"
      : frame === undefined
        ? "Waiting"
        : stale
          ? "Stale"
          : "Live";
  const indicatorClass = indicator === "Live" ? "is-live" : indicator === "Stale" ? "is-stale" : "is-offline";
  const Icon = kind === "screen" ? Monitor : Map;
  const title = kind === "screen" ? "Agent viewport" : "Minimap";

  return (
    <section className={`frame-card frame-${kind}`} aria-label={title}>
      <div className="frame-header">
        <div className="frame-title">
          <Icon size={15} aria-hidden="true" />
          <span>{title}</span>
          <span className={`live-indicator ${indicatorClass}`}>
            {indicator}
          </span>
        </div>
        <div className="frame-meta mono">
          {frame?.game_loop !== undefined && <span>loop {frame.game_loop}</span>}
          {age !== null && <span>{age < 1 ? "<1s" : `${Math.floor(age)}s`} old</span>}
          {frame?.width && frame?.height && <span>{`${frame.width}×${frame.height}`}</span>}
        </div>
      </div>
      <div className="frame-canvas">
        {frame && !imageFailed ? (
          <img
            src={frameUrl(kind, frame.frame_sequence)}
            alt={kind === "screen" ? "Current StarCraft II agent RGB viewport" : "Current StarCraft II RGB minimap"}
            onError={() => setFailedSequence(frame.frame_sequence)}
          />
        ) : (
          <div className="frame-empty">
            <ImageOff size={kind === "screen" ? 38 : 24} aria-hidden="true" />
            <strong>{historical ? "Historical RGB unavailable" : imageFailed ? "Frame unavailable" : "Waiting for RGB stream"}</strong>
            <span>
              {historical
                ? "Historical RGB unavailable: frames were not persisted."
                : imageFailed
                  ? "The match continues while the image channel recovers."
                  : "No frame has been received for this session."}
            </span>
          </div>
        )}
        {kind === "screen" && frame && (
          <div className="frame-corner" title="The image is fit without interpolation or fabricated frames.">
            <Expand size={13} aria-hidden="true" /> Actual agent view
          </div>
        )}
      </div>
    </section>
  );
}

interface FramePanelProps {
  screen?: FrameMetadata;
  minimap?: FrameMetadata;
  connection: ConnectionStatus;
  staleAfterSeconds?: number;
  historical?: boolean;
}

export function FramePanel({ screen, minimap, connection, staleAfterSeconds = 2, historical = false }: FramePanelProps) {
  return (
    <div className="frame-stage">
      <FrameView kind="screen" frame={screen} connection={connection} staleAfterSeconds={staleAfterSeconds} historical={historical} />
      <FrameView kind="minimap" frame={minimap} connection={connection} staleAfterSeconds={staleAfterSeconds} historical={historical} />
    </div>
  );
}
