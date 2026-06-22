// E2E tool coverage test: drives a real browser against a running seal instance.
//
//   1. open seal in a fresh empty chat
//   2. run scenarios for each individual tool and each pair of tools
//   3. approve every bash execution the agent requests
//   4. verify the expected tools completed and recognizable outputs rendered
//
// Setup (once): pnpm install && pnpm run install-browser
// Run against a live server (default http://localhost:3000):
//   pnpm test
//   HEADED=1 SEAL_URL=http://localhost:3000 node run-ls-date.mjs
//
// Diagnostics: per-step logs, periodic state snapshots, and screenshots written
// to /tmp/seal-e2e-*.png. Exits 0 on success, non-zero on failure.

import { chromium } from "playwright";

const SEAL_URL = process.env.SEAL_URL ?? "http://localhost:3000";
const HEADED = !!process.env.HEADED;
const HEALTH_URL = new URL("/api/health", SEAL_URL).toString();

const TIMEOUT_MS = {
  action: 5_000,
  navigation: 5_000,
  appReady: 5_000,
  emptyState: 5_000,
  promptVisible: 5_000,
  stalledProgress: 30_000,
  heartbeat: 1_000,
  afterApproval: 250,
  afterCardExpand: 150,
  stableIdle: 1_000,
  poll: 250,
  headedPause: 3_000,
};

const log = (msg) => console.log(`- ${msg}`);
const fail = (msg) => {
  console.error(`\nX ${msg}`);
  process.exitCode = 1;
};

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const safeName = (value) => value.replace(/[^a-z0-9]+/gi, "-").toLowerCase();

function bashPrompt(command) {
  return `Use exactly one tool call: bash. Run this exact command: ${command}. Do not call web_fetch or subagent. After the tool returns, answer briefly.`;
}

function webFetchPrompt(url) {
  return `Use exactly one tool call: web_fetch. Fetch this exact URL: ${url}. Do not call bash or subagent. After the tool returns, answer briefly.`;
}

function subagentPrompt(name, first, second) {
  return (
    `Use exactly one tool call: subagent. Name it "${name}" and ask it to ` +
    `reply with exactly the concatenation of "${first}" and "${second}". ` +
    "Do not call bash or web_fetch. After the tool returns, answer briefly."
  );
}

const SCENARIOS = [
  {
    name: "bash",
    prompt: bashPrompt("printf 'seal-e2e-bash-%s\\n' single"),
    tools: ["bash"],
    approvals: 1,
    outputTexts: ["seal-e2e-bash-single"],
  },
  {
    name: "web_fetch",
    prompt: webFetchPrompt(HEALTH_URL),
    tools: ["web_fetch"],
    approvals: 0,
    outputTexts: ["HTTP 200", "status", "ok"],
  },
  {
    name: "subagent",
    prompt: subagentPrompt("single-helper", "seal-e2e-subagent", "-single"),
    tools: ["subagent"],
    approvals: 0,
    outputTexts: ["seal-e2e-subagent-single"],
  },
  {
    name: "bash+bash",
    prompt:
      "Use exactly two tool calls, both bash. " +
      "First run: printf 'seal-e2e-bash-pair-%s\\n' a. " +
      "Second run: printf 'seal-e2e-bash-pair-%s\\n' b. " +
      "Do not call web_fetch or subagent. After both tools return, answer briefly.",
    tools: ["bash", "bash"],
    approvals: 2,
    outputTexts: ["seal-e2e-bash-pair-a", "seal-e2e-bash-pair-b"],
  },
  {
    name: "bash+web_fetch",
    prompt:
      "Use exactly two tool calls: first bash, then web_fetch. " +
      "For bash, run: printf 'seal-e2e-bash-%s\\n' web. " +
      `For web_fetch, fetch this exact URL: ${HEALTH_URL}. ` +
      "Do not call subagent. After both tools return, answer briefly.",
    tools: ["bash", "web_fetch"],
    approvals: 1,
    outputTexts: ["seal-e2e-bash-web", "HTTP 200", "status", "ok"],
  },
  {
    name: "bash+subagent",
    prompt:
      "Use exactly two tool calls: first bash, then subagent. " +
      "For bash, run: printf 'seal-e2e-bash-%s\\n' subagent. " +
      'For subagent, name it "bash-subagent-helper" and ask it to reply with exactly the concatenation of "seal-e2e-subagent" and "-with-bash". ' +
      "Do not call web_fetch. After both tools return, answer briefly.",
    tools: ["bash", "subagent"],
    approvals: 1,
    outputTexts: ["seal-e2e-bash-subagent", "seal-e2e-subagent-with-bash"],
  },
  {
    name: "web_fetch+web_fetch",
    prompt:
      "Use exactly two tool calls, both web_fetch. " +
      `First fetch this exact URL: ${HEALTH_URL}?case=first. ` +
      `Second fetch this exact URL: ${HEALTH_URL}?case=second. ` +
      "Do not call bash or subagent. After both tools return, answer briefly.",
    tools: ["web_fetch", "web_fetch"],
    approvals: 0,
    outputTexts: ["HTTP 200", "status", "ok"],
  },
  {
    name: "web_fetch+subagent",
    prompt:
      "Use exactly two tool calls: first web_fetch, then subagent. " +
      `For web_fetch, fetch this exact URL: ${HEALTH_URL}. ` +
      'For subagent, name it "web-subagent-helper" and ask it to reply with exactly the concatenation of "seal-e2e-subagent" and "-with-web". ' +
      "Do not call bash. After both tools return, answer briefly.",
    tools: ["web_fetch", "subagent"],
    approvals: 0,
    outputTexts: ["HTTP 200", "status", "ok", "seal-e2e-subagent-with-web"],
  },
  {
    name: "subagent+subagent",
    prompt:
      "Use exactly two tool calls, both subagent. " +
      'First, name it "subagent-alpha" and ask it to reply with exactly the concatenation of "seal-e2e-subagent-pair" and "-a". ' +
      'Second, name it "subagent-beta" and ask it to reply with exactly the concatenation of "seal-e2e-subagent-pair" and "-b". ' +
      "Do not call bash or web_fetch. After both tools return, answer briefly.",
    tools: ["subagent", "subagent"],
    approvals: 0,
    outputTexts: ["seal-e2e-subagent-pair-a", "seal-e2e-subagent-pair-b"],
  },
];

function expectedToolCounts(scenario) {
  return scenario.tools.reduce((counts, tool) => {
    counts[tool] = (counts[tool] ?? 0) + 1;
    return counts;
  }, {});
}

function expectedOutputCount(scenario, text) {
  if (text === "HTTP 200") {
    return scenario.tools.filter((tool) => tool === "web_fetch").length;
  }
  return 1;
}

async function snapshot(page) {
  const [approve, awaiting, completed, denied, errored, streaming, bodyText] =
    await Promise.all([
      page.getByRole("button", { name: /approve/i }).count(),
      page.getByText("Awaiting Approval").count(),
      page.getByText("Completed").count(),
      page.getByText("Denied").count(),
      page.getByText("Error").count(),
      page.getByRole("button", { name: "Stop" }).count(),
      page.locator("body").innerText().catch(() => ""),
    ]);
  return {
    approve,
    awaiting,
    completed,
    denied,
    errored,
    streaming: streaming > 0,
    bodyLength: bodyText.length,
    bodyTail: bodyText.slice(-400),
  };
}

function describeSnapshot(s) {
  return (
    `approve:${s.approve} awaiting:${s.awaiting} completed:${s.completed} ` +
    `denied:${s.denied} error:${s.errored} streaming:${s.streaming}`
  );
}

function progressSignature(s) {
  return JSON.stringify([
    s.approve,
    s.awaiting,
    s.completed,
    s.denied,
    s.errored,
    s.streaming,
    s.bodyLength,
    s.bodyTail,
  ]);
}

async function shot(page, scenario, label) {
  await page
    .screenshot({
      path: `/tmp/seal-e2e-${safeName(scenario.name)}-${label}.png`,
      fullPage: true,
    })
    .catch(() => {});
}

async function openFreshChat(page) {
  log(`opening ${SEAL_URL}`);
  await page.goto(SEAL_URL, {
    waitUntil: "domcontentloaded",
    timeout: TIMEOUT_MS.navigation,
  });

  const textarea = page.getByPlaceholder("Ask me anything...");
  await textarea.waitFor({ state: "visible", timeout: TIMEOUT_MS.appReady });
  await page
    .getByText("Start a conversation")
    .waitFor({ state: "visible", timeout: TIMEOUT_MS.emptyState });
  log("app ready in a fresh empty chat");
  return textarea;
}

async function approveAndWait(page, scenario) {
  let approvals = 0;
  let sawApprovalUI = false;
  let lastProgressAt = Date.now();
  let lastProgressSignature = "";
  let lastLog = 0;
  const expectedCompletions = scenario.tools.length;

  while (true) {
    const s = await snapshot(page);
    const now = Date.now();
    const signature = progressSignature(s);
    if (signature !== lastProgressSignature) {
      lastProgressSignature = signature;
      lastProgressAt = now;
    }
    if (s.approve > 0 || s.awaiting > 0) sawApprovalUI = true;

    if (now - lastLog > TIMEOUT_MS.heartbeat) {
      log(`...waiting (${describeSnapshot(s)})`);
      lastLog = now;
    }

    if (now - lastProgressAt > TIMEOUT_MS.stalledProgress) {
      await shot(page, scenario, "timeout");
      fail(
        `${scenario.name}: no e2e progress for ${TIMEOUT_MS.stalledProgress}ms; ` +
          `last state -> ${describeSnapshot(s)}`,
      );
      break;
    }

    if (s.approve > 0) {
      await page
        .getByRole("button", { name: /approve/i })
        .first()
        .click({ timeout: TIMEOUT_MS.action });
      approvals++;
      lastProgressAt = Date.now();
      log(`approved tool execution #${approvals}`);
      await page.waitForTimeout(TIMEOUT_MS.afterApproval);
      continue;
    }

    if (s.awaiting > 0) {
      const headers = page
        .getByRole("button")
        .filter({ hasText: "Awaiting Approval" });
      const n = await headers.count();
      let opened = 0;
      for (let i = 0; i < n; i++) {
        const h = headers.nth(i);
        if ((await h.getAttribute("data-state")) === "closed") {
          await h.click().catch(() => {});
          opened++;
        }
      }
      if (opened) {
        lastProgressAt = Date.now();
        log(`expanded ${opened} collapsed approval card(s)`);
      }
      await page.waitForTimeout(TIMEOUT_MS.afterCardExpand);
      continue;
    }

    const terminal = s.completed + s.denied + s.errored;
    if (
      !s.streaming &&
      !s.awaiting &&
      approvals >= scenario.approvals &&
      terminal >= expectedCompletions
    ) {
      await page.waitForTimeout(TIMEOUT_MS.stableIdle);
      const s2 = await snapshot(page);
      const terminal2 = s2.completed + s2.denied + s2.errored;
      if (
        !s2.streaming &&
        !s2.awaiting &&
        s2.approve === 0 &&
        terminal2 >= expectedCompletions
      ) {
        break;
      }
      continue;
    }

    await page.waitForTimeout(TIMEOUT_MS.poll);
  }

  if (scenario.approvals > 0 && !sawApprovalUI) {
    fail(`${scenario.name}: no tool approval UI ever appeared`);
  }
  if (approvals !== scenario.approvals) {
    fail(
      `${scenario.name}: approved ${approvals} tool execution(s), ` +
        `expected ${scenario.approvals}`,
    );
  } else if (approvals > 0) {
    log(`approved ${approvals} tool execution(s) total`);
  }
}

async function countToolHeaders(page, toolName) {
  return page
    .getByRole("button")
    .filter({ hasText: new RegExp(`\\b${escapeRegExp(toolName)}\\b`) })
    .count();
}

function countText(haystack, needle) {
  let count = 0;
  let start = 0;
  while (true) {
    const index = haystack.indexOf(needle, start);
    if (index === -1) return count;
    count++;
    start = index + needle.length;
  }
}

async function verifyScenario(page, scenario) {
  const final = await snapshot(page);
  log(
    `tool states -> Completed: ${final.completed}, ` +
      `Denied: ${final.denied}, Error: ${final.errored}`,
  );

  if (final.completed < scenario.tools.length) {
    fail(
      `${scenario.name}: ${final.completed} tool(s) completed, ` +
        `expected at least ${scenario.tools.length}`,
    );
  }
  if (final.denied > 0) fail(`${scenario.name}: ${final.denied} tool execution(s) were Denied`);
  if (final.errored > 0) fail(`${scenario.name}: ${final.errored} tool execution(s) Errored`);

  const toolCounts = expectedToolCounts(scenario);
  for (const [tool, expected] of Object.entries(toolCounts)) {
    const actual = await countToolHeaders(page, tool);
    if (actual !== expected) {
      fail(`${scenario.name}: saw ${actual} ${tool} tool card(s), expected ${expected}`);
    } else {
      log(`found ${expected} ${tool} tool card(s)`);
    }
  }

  const convo = await page.locator("body").innerText();
  for (const text of scenario.outputTexts) {
    const actual = countText(convo, text);
    const expected = expectedOutputCount(scenario, text);
    if (actual < expected) {
      fail(`${scenario.name}: missing expected text "${text}"`);
    } else {
      log(`found expected text "${text}"`);
    }
  }
}

async function runScenario(browser, scenario) {
  console.log(`\n== ${scenario.name} ==`);
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  page.setDefaultTimeout(TIMEOUT_MS.action);
  page.setDefaultNavigationTimeout(TIMEOUT_MS.navigation);

  try {
    const textarea = await openFreshChat(page);

    await textarea.fill(scenario.prompt);
    await page.getByRole("button", { name: "Submit" }).click();
    log(`sent prompt: "${scenario.prompt}"`);

    await page
      .getByText(scenario.prompt, { exact: false })
      .first()
      .waitFor({ state: "visible", timeout: TIMEOUT_MS.promptVisible });
    log("prompt is visible in the conversation");
    await shot(page, scenario, "after-send");

    await approveAndWait(page, scenario);
    await shot(page, scenario, "final");
    await verifyScenario(page, scenario);
  } catch (err) {
    await shot(page, scenario, "error");
    fail(`${scenario.name}: unexpected error: ${err?.stack || err}`);
  } finally {
    await context.close();
  }
}

const browser = await chromium.launch({
  headless: !HEADED,
  timeout: TIMEOUT_MS.action,
});

try {
  for (const scenario of SCENARIOS) {
    await runScenario(browser, scenario);
  }

  if (process.exitCode === 1) {
    console.error("\nX verification FAILED (see /tmp/seal-e2e-*.png)");
  } else {
    console.log("\nPASS - covered individual tools and tool pairs");
  }
} finally {
  if (HEADED) await new Promise((resolve) => setTimeout(resolve, TIMEOUT_MS.headedPause));
  await browser.close();
}
