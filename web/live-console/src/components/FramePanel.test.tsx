import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { FramePanel } from "./FramePanel";

describe("FramePanel", () => {
  it("explains why historical runs do not have RGB frames", () => {
    const markup = renderToStaticMarkup(<FramePanel connection="disconnected" historical />);
    expect(markup).toContain("Historical RGB unavailable: frames were not persisted.");
    expect(markup).toContain("Unavailable");
    expect(markup).not.toContain("No frame has been received for this session.");
  });
});
