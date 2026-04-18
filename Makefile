.PHONY: help install dev test lint fmt typecheck check migrate migration eval eval-save seed smoke smoke-cleanup diagram-states docker-build docker-up docker-down deploy clean

help:
	@echo "Available targets:"
	@echo "  install         - Install dependencies with uv"
	@echo "  dev             - Run dev server with reload"
	@echo "  test            - Run tests with coverage"
	@echo "  lint            - Lint with ruff"
	@echo "  fmt             - Format with ruff"
	@echo "  typecheck       - Type check with mypy"
	@echo "  check           - Run lint + typecheck + test"
	@echo "  migrate         - Apply Alembic migrations"
	@echo "  migration       - Create new migration (usage: make migration m='describe your change')"
	@echo "  eval            - Run evaluation framework against golden set"
	@echo "  seed            - Bulk-ingest scripts/seed_urls.txt (50 curated dev.to posts) — needs 'make dev' running"
	@echo "  smoke           - End-to-end smoke against real Qdrant+OpenRouter (needs 'make dev' in another terminal)"
	@echo "  smoke-cleanup   - Remove smoke test's points from Qdrant (usage: make smoke-cleanup URL=...)"
	@echo "  diagram-states  - Export ingestion FSM state diagram (PNG)"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-up       - Start docker-compose (app + local Qdrant)"
	@echo "  docker-down     - Stop docker-compose"
	@echo "  deploy          - Deploy to Fly.io"
	@echo "  clean           - Remove caches and build artifacts"

install:
	uv sync --all-extras

dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy app

check: lint typecheck test

migrate:
	uv run alembic upgrade head

migration:
	uv run alembic revision --autogenerate -m "$(m)"

eval:
	@APP_URL="$(APP_URL)" bash -c 'uv run python -m app.evaluation.runner $${APP_URL:+--app-url "$$APP_URL"}'

eval-save:
	# pipefail so tee doesn't mask the runner's exit code — regressions
	# must bubble up to CI / local shells. APP_URL=https://... targets prod.
	@APP_URL="$(APP_URL)" bash -o pipefail -c 'uv run python -m app.evaluation.runner $${APP_URL:+--app-url "$$APP_URL"} | tee docs/eval-results.txt'

smoke:
	uv run python scripts/smoke_ingestion.py "$(URL)"

seed:
	uv run python scripts/seed_corpus.py $(URLS)

smoke-cleanup:
	@test -n "$(URL)" || (echo "Usage: make smoke-cleanup URL=https://dev.to/..." && exit 1)
	uv run python scripts/smoke_cleanup.py "$(URL)"

diagram-states:
	uv run python -c "from app.ingestion.state import get_state_diagram; get_state_diagram('docs/diagrams/ingestion-fsm.png')"

docker-build:
	docker build -t semantic-document-search:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

deploy:
	flyctl deploy --remote-only --build-arg GIT_SHA=$$(git rev-parse HEAD)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
