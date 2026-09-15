# seal

A personal AI assistant built as a **durable agent**. Temporal owns session
and turn execution, so conversations survive worker restarts, streams can be
resumed mid-run, and tool calls can wait for human approval.

Seal is an example app for the [AI SDK for Python](https://ai-python.dev)
(the `ai` package) and [Temporal](https://temporal.io/).

The agent (Claude via the AI Gateway) has three tools: `bash`,
`web_fetch`, and `subagent`. Bash runs are gated behind an approval UI
when run by the main agent, but not when run by a subagent. (That is
silly, but this is a demo app.)

## How it works

- **frontend/** — React + Vite chat UI using the AI SDK (`useChat`) and
  [AI Elements](https://elements.ai-sdk.dev). Reconnecting to a session
  re-tails the in-flight stream (`useChat({ resume: true })`).
- **backend/app/** — FastAPI service. `POST /api/chat` starts (or resumes) a
  run and streams the AI SDK UI message protocol; other endpoints cover
  sessions, titles, and private blob attachments. See `app/server.py` for the
  endpoint list.
- **backend/agent/** — `driver.py` hosts one long-lived Temporal workflow per
  chat session. New messages, approvals, and interrupts are Temporal signals.
  Model and tool calls are Temporal activities. Subagents run as child
  workflows.
- **Streaming and state** — each session and subagent workflow hosts a Temporal
  Workflow Stream. Activities publish model events into it and FastAPI resumes
  from durable offsets. Committed message history is queried from workflow
  state. Attachments still use Vercel Blob.

The frontend and FastAPI service can still run on Vercel. The Temporal worker
must run as a separate long-lived process.

## Development

Prereqs: [uv](https://docs.astral.sh/uv/), [pnpm](https://pnpm.io), the
[Vercel CLI](https://vercel.com/docs/cli), and the
[Temporal CLI](https://docs.temporal.io/cli).

```sh
./dev-setup.sh
cd frontend && pnpm install

# terminal 1: Temporal server and UI (:8233)
temporal server start-dev

# terminal 2: Temporal worker
cd backend && uv run worker

# terminal 3: frontend + FastAPI (:3000)
vercel dev
```

Environment: `AI_GATEWAY_API_KEY` (model access), optional `TEMPORAL_ADDRESS`
(default `localhost:7233`), optional `TEMPORAL_NAMESPACE` (default `default`),
optional `DATABASE_URL` (shared session-list metadata), and a blob token for
attachments. Temporal Workflow Streams are currently experimental.

### Checks

```sh
make ci               # everything below
make ci-backend       # uv sync, ruff, mypy, ty, pytest
make ci-frontend      # pnpm install, prettier, eslint, tsc, vitest, build
```

### E2E tests

`e2e/` drives a real browser against a running instance:

```sh
cd e2e && pnpm install && pnpm run install-browser
pnpm test             # expects the app at http://localhost:3000
pnpm run test:images  # image latency: time to first image, time to all N
```

`test:images` prompts "draw N pictures of things you find interesting"
(`N=5` by default) and reports when each image actually painted, measured
from the submit click. Timings also land in
`/tmp/seal-e2e-images-summary.json`.

## Deployment

Deploy the frontend and FastAPI project with `vc deploy`, and run
`cd backend && uv run worker` on a long-lived host that can reach the same
Temporal namespace. Set `DATABASE_URL` for FastAPI when multiple instances need
to share the session list; agent state and SSE offsets live in Temporal.
