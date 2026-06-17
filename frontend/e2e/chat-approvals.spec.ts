import { expect, test, type Page } from "@playwright/test";
import childProcess from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

type ScriptMessage = {
  text: string;
  tools?: Array<{ id: string; name: string; args: Record<string, unknown> }>;
};

type Script = {
  responses: ScriptMessage[][];
  keyed_responses?: Record<string, ScriptMessage[]>;
};

type RunningApp = {
  baseUrl: string;
  proc: childProcess.ChildProcessWithoutNullStreams;
  tmp: string;
  output: () => string;
};

async function freePort(): Promise<number> {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("could not allocate a port"));
        return;
      }
      server.close(() => resolve(address.port));
    });
  });
}

async function until(
  condition: () => boolean | Promise<boolean>,
  message: () => string,
  timeoutMs = 30_000,
): Promise<void> {
  const started = Date.now();
  while (!(await condition())) {
    if (Date.now() - started > timeoutMs) {
      throw new Error(message());
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function startApp(script: Script): Promise<RunningApp> {
  const port = await freePort();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "seal-playwright-"));
  const scriptPath = path.join(tmp, "model.json");
  fs.writeFileSync(scriptPath, JSON.stringify(script), "utf8");

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    SEAL_TEST_MODEL_SCRIPT: scriptPath,
    SEAL_STREAMS_DIR: path.join(tmp, "streams"),
    SEAL_SESSIONS_DIR: path.join(tmp, "sessions"),
    WORKFLOW_LOCAL_DATA_DIR: path.join(tmp, "workflow"),
  };
  delete env.DATABASE_URL;

  let stdout = "";
  let stderr = "";
  const proc = childProcess.spawn(
    "vercel",
    ["dev", "--listen", `127.0.0.1:${port}`, "--yes", "--local"],
    { cwd: ROOT, env },
  );
  proc.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });
  proc.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  const app = {
    baseUrl: `http://127.0.0.1:${port}`,
    proc,
    tmp,
    output: () => `${stdout}\n${stderr}`.trim(),
  };

  await until(
    async () => {
      if (proc.exitCode !== null) {
        throw new Error(`vercel dev exited early:\n${app.output()}`);
      }
      try {
        const response = await fetch(`${app.baseUrl}/api/health`);
        return response.ok;
      } catch {
        return false;
      }
    },
    () => `vercel dev did not become ready:\n${app.output()}`,
  );

  return app;
}

async function stopApp(app: RunningApp): Promise<void> {
  if (app.proc.exitCode === null) {
    app.proc.kill("SIGTERM");
    await Promise.race([
      new Promise((resolve) => app.proc.once("exit", resolve)),
      new Promise((resolve) => setTimeout(resolve, 2_000)),
    ]);
  }
  fs.rmSync(app.tmp, { recursive: true, force: true });
}

async function openApp(page: Page, app: RunningApp): Promise<void> {
  await page.goto(app.baseUrl);
  await expect(page.getByPlaceholder("Ask me anything...")).toBeVisible();
}

async function sendPrompt(page: Page, prompt: string): Promise<void> {
  await page.getByPlaceholder("Ask me anything...").fill(prompt);
  await page.getByRole("button", { name: "Submit" }).click();
}

async function expandPendingApprovalTools(
  page: Page,
  expectedCount: number,
): Promise<void> {
  const pendingTools = page.getByRole("button", { name: /Awaiting Approval/ });
  await expect(pendingTools).toHaveCount(expectedCount);

  for (let index = 0; index < expectedCount; index += 1) {
    const tool = pendingTools.nth(index);
    if ((await tool.getAttribute("aria-expanded")) !== "true") {
      await tool.click();
    }
  }
}

async function approveFirst(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Approve" }).first().click();
}

async function rejectFirst(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Reject" }).first().click();
}

async function sessionHistory(page: Page): Promise<string> {
  const sessionsResponse = await page.request.get("/api/sessions");
  expect(sessionsResponse.ok()).toBeTruthy();
  const sessions = (await sessionsResponse.json()) as Array<{ id: string }>;
  expect(sessions.length).toBe(1);

  const response = await page.request.get(`/api/sessions/${sessions[0].id}`);
  expect(response.ok()).toBeTruthy();
  return JSON.stringify(await response.json());
}

function parallelApprovalsScript(): Script {
  return {
    responses: [
      [
        {
          text: "running both",
          tools: [
            { id: "tc-a", name: "bash", args: { command: "echo alpha-out" } },
            { id: "tc-b", name: "bash", args: { command: "echo beta-out" } },
          ],
        },
      ],
      [{ text: "both handled" }],
      [{ text: "run both title" }],
    ],
  };
}

function mixedSubagentApprovalScript(): Script {
  return {
    responses: [
      [
        {
          text: "delegating and running",
          tools: [
            {
              id: "tc-sub",
              name: "subagent",
              args: { prompt: "task-gamma", name: "gamma" },
            },
            { id: "tc-cmd", name: "bash", args: { command: "echo gamma-out" } },
          ],
        },
      ],
      [{ text: "wrapped up" }],
      [{ text: "mixed title" }],
    ],
    keyed_responses: {
      "task-gamma": [{ text: "gamma report" }],
    },
  };
}

test("bash + bash approval resumes through the real app", async ({ page }) => {
  const app = await startApp(parallelApprovalsScript());
  try {
    await openApp(page, app);
    await sendPrompt(page, "run both commands");

    await expandPendingApprovalTools(page, 2);
    await expect(page.getByRole("button", { name: "Approve" })).toHaveCount(2);
    await expect(page.getByRole("button", { name: "Reject" })).toHaveCount(2);

    await approveFirst(page);
    await rejectFirst(page);

    await expect(page.getByText("both handled")).toBeVisible();
    const history = await sessionHistory(page);
    expect(history).toContain("alpha-out");
    expect(history).toContain("Rejected");
  } finally {
    await stopApp(app);
  }
});

test("subagent + bash approval resumes through the real app", async ({
  page,
}) => {
  const app = await startApp(mixedSubagentApprovalScript());
  try {
    await openApp(page, app);
    await sendPrompt(page, "delegate and run");

    await expandPendingApprovalTools(page, 1);
    await expect(page.getByRole("button", { name: "Approve" })).toHaveCount(1);

    await approveFirst(page);

    await expect(page.getByText("wrapped up")).toBeVisible();
    const history = await sessionHistory(page);
    expect(history).toContain("gamma-out");
    expect(history).toContain("gamma report");
  } finally {
    await stopApp(app);
  }
});

test("reload while parked on approval can still finish", async ({ page }) => {
  const app = await startApp(mixedSubagentApprovalScript());
  try {
    await openApp(page, app);
    await sendPrompt(page, "delegate and run");

    await expandPendingApprovalTools(page, 1);
    await expect(page.getByRole("button", { name: "Approve" })).toHaveCount(1);
    await page.reload();
    await expect(page.getByPlaceholder("Ask me anything...")).toBeVisible();

    await expandPendingApprovalTools(page, 1);
    await expect(page.getByRole("button", { name: "Approve" })).toHaveCount(1);
    await approveFirst(page);

    await expect(page.getByText("wrapped up")).toBeVisible();
    const history = await sessionHistory(page);
    expect(history).toContain("gamma-out");
    expect(history).toContain("gamma report");
  } finally {
    await stopApp(app);
  }
});
