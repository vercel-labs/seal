.PHONY: ci ci-backend ci-frontend \
       backend-sync backend-format backend-lint backend-typecheck backend-ty backend-test \
       frontend-install frontend-format frontend-lint frontend-typecheck frontend-build

# Run all CI checks
ci: ci-backend ci-frontend

# --- Backend --------------------------------------------------------------- #

ci-backend: backend-sync backend-format backend-lint backend-typecheck backend-ty backend-test

backend-sync:
	cd backend && uv sync

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

ci-frontend: frontend-install frontend-format frontend-lint frontend-typecheck frontend-build

frontend-install:
	cd frontend && pnpm install --frozen-lockfile

frontend-format:
	cd frontend && pnpm run format:check

frontend-lint:
	cd frontend && pnpm run lint

frontend-typecheck:
	cd frontend && pnpm run typecheck

frontend-build:
	cd frontend && pnpm run build
