// E2E smoke test: drives a real browser against a running seal instance.
//
//   1. open seal in a fresh empty chat
//   2. prompt "run ls and date"
//   3. approve every tool execution the agent requests (expects two: ls + date)
//   4. verify both bash commands ran and the agent produced a final reply
//
// Setup (once): pnpm install && pnpm run install-browser
// Run against a live server (default http://localhost:3000):
//   pnpm test
//   HEADED=1 SEAL_URL=http://localhost:3000 node run-ls-date.mjs
//
// Diagnostics: per-step logs, periodic state snapshots, and screenshots written
// to /tmp/seal-e2e-*.png. Exits 0 on success, non-zero on failure.

import { chromium } from "playwright";

const URL = process.env.SEAL_URL ?? "http://localhost:3000";
const HEADED = !!process.env.HEADED;
const PROMPT = "run ls and date";
const YEAR = String(new Date().getFullYear());

const log = (msg) => console.log(`• ${msg}`);
const fail = (msg) => {
  console.error(`\n✗ ${msg}`);
  process.exitCode = 1;
};

const browser = await chromium.launch({ headless: !HEADED });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const shot = (label) =>
  page.screenshot({ path: `/tmp/seal-e2e-${label}.png`, fullPage: true }).catch(() => {});

// Count what the script currently perceives -- used for live diagnostics.
async function snapshot() {
  const [approve, awaiting, completed, denied, errored, streaming] =
    await Promise.all([
      page.getByRole("button", { name: /approve/i }).count(),
      page.getByText("Awaiting Approval").count(),
      page.getByText("Completed").count(),
      page.getByText("Denied").count(),
      page.getByText("Error").count(),
      page.getByRole("button", { name: "Stop" }).count(),
    ]);
  return { approve, awaiting, completed, denied, errored, streaming: streaming > 0 };
}

try {
  // --- 1. open seal in a fresh empty chat --------------------------------
  log(`opening ${URL}`);
  await page.goto(URL, { waitUntil: "domcontentloaded" });

  // A fresh browser context has empty localStorage, so bootstrap() drops us
  // straight into a new empty chat -- no need to click "New chat". App shows
  // "Loading..." until bootstrap() finishes and ChatView mounts; the textarea
  // only exists once it's up, and the empty-state confirms a clean conversation.
  const textarea = page.getByPlaceholder("Ask me anything...");
  await textarea.waitFor({ state: "visible", timeout: 30_000 });
  await page
    .getByText("Start a conversation")
    .waitFor({ state: "visible", timeout: 15_000 });
  log("app ready in a fresh empty chat");

  // --- 2. send the prompt -------------------------------------------------
  await textarea.fill(PROMPT);
  await page.getByRole("button", { name: "Submit" }).click();
  log(`sent prompt: "${PROMPT}"`);

  // Confirm the prompt actually landed in the *visible* conversation (catches
  // any session/remount race -- if this fails, we'd be staring at the wrong UI).
  await page
    .getByText(PROMPT, { exact: false })
    .first()
    .waitFor({ state: "visible", timeout: 15_000 });
  log("prompt is visible in the conversation");
  await shot("after-send");

  // --- 3. approve each tool execution as it appears ----------------------
  // bash requires approval, so each command surfaces a Confirmation with an
  // Approve button. Approvals may arrive in parallel or one per turn; keep
  // approving until the agent goes idle.
  let approvals = 0;
  let sawApprovalUI = false;
  const start = Date.now();
  const MAX_MS = 120_000;
  let lastLog = 0;

  while (Date.now() - start < MAX_MS) {
    const s = await snapshot();
    if (s.approve > 0 || s.awaiting > 0) sawApprovalUI = true;

    // Throttled heartbeat so we can see what the script perceives.
    if (Date.now() - lastLog > 2500) {
      log(
        `…waiting (approve:${s.approve} awaiting:${s.awaiting} ` +
          `completed:${s.completed} streaming:${s.streaming})`,
      );
      lastLog = Date.now();
    }

    if (s.approve > 0) {
      await page.getByRole("button", { name: /approve/i }).first().click();
      approvals++;
      log(`approved tool execution #${approvals}`);
      await page.waitForTimeout(600); // let state settle / auto-send fire
      continue;
    }

    // Approval requested but no Approve button is reachable: the tool card
    // mounted while "Running" (defaultOpen=false) and Radix only honors
    // defaultOpen at mount, so it stays collapsed after flipping to
    // "Awaiting Approval", hiding its Approve button. Expand the card(s).
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
      if (opened) log(`expanded ${opened} collapsed approval card(s)`);
      await page.waitForTimeout(300);
      continue;
    }

    // No pending approval. We're only done once every approved tool has reached
    // a terminal state (Completed/Denied/Error) AND the agent stopped streaming.
    // Requiring terminal tools avoids bailing in the brief idle gap between
    // turns, before the bash output (and final reply) has rendered.
    const terminal = s.completed + s.denied + s.errored;
    if (!s.streaming && !s.awaiting && approvals >= 1 && terminal >= approvals) {
      await page.waitForTimeout(1500);
      const s2 = await snapshot();
      const terminal2 = s2.completed + s2.denied + s2.errored;
      if (!s2.streaming && !s2.awaiting && s2.approve === 0 && terminal2 >= approvals) {
        break;
      }
      continue;
    }

    await page.waitForTimeout(500);
  }

  await shot("final");

  if (!sawApprovalUI) fail("no tool approval UI ever appeared");
  if (approvals === 0) fail("no tool approval was ever clicked");
  else log(`approved ${approvals} tool execution(s) total`);

  // --- 4. verify it worked ------------------------------------------------
  const final = await snapshot();
  log(
    `tool states -> Completed: ${final.completed}, ` +
      `Denied: ${final.denied}, Error: ${final.errored}`,
  );

  if (final.completed < 1) fail("no tool reached the Completed state");
  if (final.denied > 0) fail(`${final.denied} tool execution(s) were Denied`);
  if (final.errored > 0) fail(`${final.errored} tool execution(s) Errored`);

  // The whole conversation as plain text: should contain both commands and the
  // date command's output (the current year).
  const convo = await page.locator("body").innerText();

  if (!/\bls\b/.test(convo)) fail('conversation has no "ls" command');
  else log('found "ls" in the conversation');

  if (!/\bdate\b/.test(convo)) fail('conversation has no "date" command');
  else log('found "date" in the conversation');

  if (!convo.includes(YEAR)) fail(`date output missing current year (${YEAR})`);
  else log(`found current year (${YEAR}) -- date ran`);

  if (process.exitCode === 1) {
    console.error("\n✗ verification FAILED (see /tmp/seal-e2e-*.png)");
  } else {
    console.log("\n✓ PASS -- ran ls + date, approved executions, verified output");
  }
} catch (err) {
  await shot("error");
  fail(`unexpected error: ${err?.stack || err}`);
} finally {
  if (HEADED) await page.waitForTimeout(3000);
  await browser.close();
}
