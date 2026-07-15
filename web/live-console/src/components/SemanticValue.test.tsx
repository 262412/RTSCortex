import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { SemanticValue } from "./SemanticValue";

describe("SemanticValue", () => {
  it("renders protocol objects as readable labels and values", () => {
    const markup = renderToStaticMarkup(
      <SemanticValue
        value={{
          action_name: "Build_Pylon_Screen",
          actor: "Builder/Builder-Probe-1",
          status: "succeeded",
          command_id: "cmd-pylon-1",
          requested_arguments: [[66, 88]],
        }}
      />,
    );

    expect(markup).toContain("建造水晶塔（Build_Pylon_Screen）");
    expect(markup).toContain("建造工编队（Builder/Builder-Probe-1）");
    expect(markup).toContain("状态");
    expect(markup).toContain("成功（succeeded）");
    expect(markup).toContain("动作 ID");
    expect(markup).toContain("cmd-pylon-1");
    expect(markup).not.toContain("&quot;action_name&quot;");
  });

  it("preserves model prose as prose instead of translating its meaning", () => {
    const reflection = "The last action succeeded, but it did not advance the strategic goal.";
    const markup = renderToStaticMarkup(<SemanticValue value={{ reflection, lessons: ["Build a Pylon first."] }} />);

    expect(markup).toContain(reflection);
    expect(markup).toContain("得到的经验");
    expect(markup).toContain("Build a Pylon first.");
  });
});

