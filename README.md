# seal

A personal AI assistant built as a **durable agent**: every agent run is a
Vercel workflow, so turns survive restarts, streams can be resumed mid-run,
and tool calls can park indefinitely waiting for human approval.

Seal is an example app for the [AI SDK for Python](https://ai-python.dev)
(the `ai` package) and for
[Workflows with Python](https://vercel.com/docs/workflows/python)
(`vercel.workflow`).

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
- **backend/agent/** — the durable agent itself. `driver.py` runs a
  `run_session` workflow that spawns one child `run_turn` workflow per agent
  turn and suspends on a hook until it finishes. Tool approvals are workflow
  hooks too: the turn parks until the user answers, then resumes with the
  decision. Model calls, stream writes, and session snapshots are all
  workflow steps, replay-safe via the workflow's deterministic RNG/clock.
- **Storage** (`agent/storage.py`) — one append-only primitive backing
  both durable streams and session snapshots. Uses Postgres when
  `DATABASE_URL` is set, local jsonl files otherwise. Uses Vercel Blob
  to store attachments when available.

Deployment is two Vercel services (see `vercel.json`): the frontend and the
backend, with the workflow worker declared in `backend/pyproject.toml`.

## Development

Prereqs: [uv](https://docs.astral.sh/uv/), [pnpm](https://pnpm.io), and the
[Vercel CLI](https://vercel.com/docs/cli).

```sh
./dev-setup.sh        # sync backend deps (works around a vercel-worker version override)
cd frontend && pnpm install
vercel dev            # serves frontend + backend + worker on :3000
```

Environment: `AI_GATEWAY_API_KEY` (model access), optional `DATABASE_URL`
(Postgres storage), and a blob token for attachments.

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
```

## Deployment

Deploy as a project to Vercel with `vc deploy`. `DATABASE_URL` must
point to a Postgres database, which can most easily be done by
configuring a marketplace integration with Neon or similar.
