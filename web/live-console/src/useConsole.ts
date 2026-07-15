import { useEffect, useReducer, useRef } from "react";

import { ConsoleStream, fetchEvents, fetchSession } from "./api";
import { consoleReducer, createInitialState } from "./data";
import type { ConsoleMessage, StoredEvent } from "./types";

export function useConsole() {
  const [state, dispatch] = useReducer(consoleReducer, undefined, createInitialState);
  const lastEventIdRef = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    let stream: ConsoleStream | null = null;

    const addEvents = (events: StoredEvent[]) => {
      if (events.length > 0) {
        lastEventIdRef.current = Math.max(lastEventIdRef.current, ...events.map((event) => event.event_id));
        dispatch({ type: "events", events });
      }
    };

    const resyncFrom = (afterEventId: number) => {
      void fetchEvents(afterEventId, state.eventLimit, controller.signal)
        .then(addEvents)
        .catch((error: unknown) => {
          if (!controller.signal.aborted) {
            dispatch({ type: "error", error: error instanceof Error ? error.message : "Resync failed" });
          }
        });
    };

    const handleMessage = (message: ConsoleMessage) => {
      switch (message.type) {
        case "stored_event":
          lastEventIdRef.current = Math.max(lastEventIdRef.current, message.event.event_id);
          dispatch({ type: "event", event: message.event });
          break;
        case "frame_available":
          dispatch({ type: "frame", frame: message.frame });
          break;
        case "run_status":
          dispatch({ type: "session", session: message.session });
          break;
        case "heartbeat":
          dispatch({ type: "heartbeat" });
          if (message.latest_event_id !== undefined && message.latest_event_id > lastEventIdRef.current) {
            resyncFrom(lastEventIdRef.current);
          }
          break;
        case "resync_required":
          resyncFrom(Math.max(lastEventIdRef.current, message.after_event_id ?? 0));
          break;
      }
    };

    const bootstrap = async () => {
      try {
        const session = await fetchSession(controller.signal);
        if (controller.signal.aborted) return;
        dispatch({ type: "session", session });
        const eventLimit = session.frontend_event_limit ?? state.eventLimit;
        addEvents(await fetchEvents(0, eventLimit, controller.signal));
        if (controller.signal.aborted) return;
        stream = new ConsoleStream({
          getAfterEventId: () => lastEventIdRef.current,
          onMessage: handleMessage,
          onStatus: (status) => dispatch({ type: "connection", status }),
          onBackfill: addEvents,
          onError: (error) => dispatch({ type: "error", error }),
        });
        await stream.start();
      } catch (error) {
        if (!controller.signal.aborted) {
          dispatch({ type: "connection", status: "disconnected" });
          dispatch({ type: "error", error: error instanceof Error ? error.message : "Console bootstrap failed" });
        }
      }
    };

    void bootstrap();
    return () => {
      controller.abort();
      stream?.stop();
    };
    // The console owns one connection for its full mounted lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { state, dispatch };
}
