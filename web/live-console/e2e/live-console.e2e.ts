import { expect, test, type WebSocketRoute } from "@playwright/test";

test("historical console renders, reconnects without duplicates, and stays read-only", async ({ page, request }) => {
  let activeSocket: WebSocketRoute | undefined;
  let rejectSockets = false;
  await page.routeWebSocket(/\/console\/api\/v1\/stream/, async (socket) => {
    if (rejectSockets) {
      await socket.close({ code: 1012, reason: "E2E reconnect window" });
      return;
    }
    activeSocket = socket;
    socket.connectToServer();
  });
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "RTSCortex" })).toBeVisible();
  await expect(page.locator(".connection-pill")).toHaveText("Live");
  await expect(page.locator(".run-state")).toHaveText("historical");
  await expect(page.locator(".status-header")).toContainText("e2e-console-run");
  await expect(page.locator(".status-header")).toContainText("Simple64 / 0");
  await expect(page.locator(".status-header")).toContainText("Qwen/Qwen3-8B");
  await expect(page.getByText("Historical RGB unavailable: frames were not persisted.", { exact: true })).toHaveCount(2);
  await expect(page.locator(".console-footer")).toContainText("9 / 5,000 events retained");
  await expect(page.getByRole("heading", { name: "当前采用的计划" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "模型建议动作" })).toBeVisible();
  await expect(page.locator(".decision-rail")).toContainText("Establish Protoss production");
  await expect(page.locator(".decision-rail")).toContainText("建造水晶塔（Build_Pylon_Screen）");

  const desktopFrame = await page.locator(".frame-stage").boundingBox();
  const desktopRail = await page.locator(".decision-rail").boundingBox();
  expect(desktopFrame).not.toBeNull();
  expect(desktopRail).not.toBeNull();
  expect(desktopRail!.x).toBeGreaterThanOrEqual(desktopFrame!.x + desktopFrame!.width - 1);

  expect(activeSocket).toBeDefined();
  rejectSockets = true;
  await activeSocket!.close({ code: 1012, reason: "E2E reconnect window" });
  await expect(page.locator(".connection-pill")).toHaveText("Reconnecting");
  await page.waitForTimeout(3_000);
  rejectSockets = false;
  await expect(page.locator(".connection-pill")).toHaveText("Live", { timeout: 10_000 });
  await expect(page.locator(".console-footer")).toContainText("12 / 5,000 events retained");

  const sessionResponse = await request.get("/console/api/v1/session");
  expect(sessionResponse.ok()).toBeTruthy();
  const sessionSnapshot = (await sessionResponse.json()) as { latest_event_id: number };
  expect(sessionSnapshot.latest_event_id).toBe(12);

  const executionRow = page.locator(".event-row").filter({
    has: page.locator(".event-type", { hasText: /^动作执行结果$/ }),
  });
  await executionRow.click();
  const drawer = page.getByRole("dialog", { name: "动作执行结果" });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByText("动作完整生命周期")).toBeVisible();
  await expect(drawer.getByText("3 条事件", { exact: true })).toBeVisible();
  await expect(drawer).toContainText("cmd-build-pylon-001");
  await expect(drawer.locator(".json-detail")).toBeHidden();
  await page.keyboard.press("Escape");
  await expect(drawer).toBeHidden();

  await page.getByRole("button", { name: "失败", exact: true }).click();
  await expect(page.locator(".event-count")).toHaveText("1 条事件");
  await expect(page.locator(".event-row")).toContainText("目标属于己方");
  await page.locator(".event-row").click();
  const failedDrawer = page.getByRole("dialog", { name: "动作验证失败" });
  await expect(failedDrawer.locator(".json-detail")).toBeHidden();
  await failedDrawer.getByText("技术详情：查看原始 JSON").click();
  await expect(failedDrawer.locator(".json-detail")).toBeVisible();
  await expect(failedDrawer.locator(".json-detail")).toContainText("friendly_target");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "全部", exact: true }).click();

  await page.setViewportSize({ width: 1_024, height: 900 });
  const tabletFrame = await page.locator(".frame-stage").boundingBox();
  const tabletRail = await page.locator(".decision-rail").boundingBox();
  expect(tabletFrame).not.toBeNull();
  expect(tabletRail).not.toBeNull();
  expect(tabletRail!.y).toBeGreaterThanOrEqual(tabletFrame!.y + tabletFrame!.height - 1);

  const healthResponse = await request.get("/console/api/v1/health");
  expect(healthResponse.ok()).toBeTruthy();
  expect(await healthResponse.json()).toMatchObject({ status: "ok", read_only: true });
  const controlResponse = await request.post("/v1/tick", { data: {} });
  expect([404, 405]).toContain(controlResponse.status());
});
