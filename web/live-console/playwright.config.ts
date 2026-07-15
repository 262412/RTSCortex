import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.RTSCORTEX_E2E_PORT ?? "8877");
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.e2e.ts",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: "line",
  outputDir: "../../output/playwright",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: executablePath ? { executablePath } : undefined,
      },
    },
  ],
  webServer: {
    command: [
      "uv run python e2e/console_fixture.py",
      `--port ${port}`,
      "--static-dir ../../src/rtscortex/console/static",
    ].join(" "),
    url: `http://127.0.0.1:${port}/console/api/v1/health`,
    reuseExistingServer: false,
    timeout: 30_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
