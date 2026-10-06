.PHONY: ci ci-backend ci-frontend \
       backend-sync backend-upgrade-ai backend-upgrade-vercel backend-format backend-lint backend-typecheck backend-ty backend-test \
       frontend-install frontend-format frontend-lint frontend-typecheck frontend-test frontend-build

# Run all CI checks
ci: ci-backend ci-frontend

# --- Backend --------------------------------------------------------------- #

ci-backend: backend-sync backend-format backend-lint backend-typecheck backend-ty backend-test

backend-sync:
	cd backend && uv sync

backend-upgrade-ai:
	cd backend && uv lock --upgrade-package ai

# vercel-py ships several packages from one repo; they must move together.
backend-upgrade-vercel:
	cd backend && uv lock \
		--upgrade-package vercel \
		--upgrade-package vercel-cache \
		--upgrade-package vercel-headers \
		--upgrade-package vercel-internal-core \
		--upgrade-package vercel-internal-telemetry \
		--upgrade-package vercel-oidc \
		--upgrade-package vercel-queue \
		--upgrade-package vercel-sandbox \
		--upgrade-package vercel-workflow

backend-format:
	cd backend && uv run ruff format --check .

backend-lint:
	cd backend && uv run ruff check .

backend-typecheck:
	cd backend && uv run mypy .

backend-ty:
	cd backend && uv run ty check

backend-test:
	cd backend && uv run pytest || test $$? -eq 5

# --- Frontend -------------------------------------------------------------- #

ci-frontend: frontend-install frontend-format frontend-lint frontend-typecheck frontend-test frontend-build

frontend-install:
	cd frontend && pnpm install --frozen-lockfile

frontend-format:
	cd frontend && pnpm run format:check

frontend-lint:
	cd frontend && pnpm run lint

frontend-typecheck:
	cd frontend && pnpm run typecheck

frontend-test:
	cd frontend && pnpm run test

frontend-build:
	cd frontend && pnpm run build
