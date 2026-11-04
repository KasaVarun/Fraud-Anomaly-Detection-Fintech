UV ?= uv

.PHONY: setup format format-check lint type-check test test-schema check run \
	db-up db-down db-status db-test-create db-migrate db-downgrade db-reset \
	gate-a-preflight bicep-validate cost-review container-build container-inspect \
	worker-build worker-inspect contracts-check secret-scan

setup:
	$(UV) sync --all-groups

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

format-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

type-check:
	$(UV) run mypy src tests

test:
	$(UV) run pytest --cov=fraud_detection --cov-report=term-missing

test-schema:
	$(UV) run pytest tests/schema tests/integration -m "schema or integration"

check: format-check lint type-check test

run:
	$(UV) run uvicorn fraud_detection.api.app:app --reload --host 127.0.0.1 --port 8000

gate-a-preflight:
	$(UV) run fraud-deploy preflight

bicep-validate:
	DOTNET_BUNDLE_EXTRACT_BASE_DIR=$(CURDIR)/.tools/cache/bicep $(CURDIR)/.tools/bin/bicep build infra/main.bicep --stdout >/dev/null

cost-review:
	$(UV) run fraud-deploy cost-review

container-build:
	$(UV) run fraud-deploy container-build --tag fraud-investigator:gate-a-amd64

container-inspect:
	$(UV) run fraud-deploy container-inspect --tag fraud-investigator:gate-a-amd64

worker-build:
	$(UV) run fraud-deploy worker-build --tag fraud-worker:gate-a-amd64

worker-inspect:
	$(UV) run fraud-deploy worker-inspect --tag fraud-worker:gate-a-amd64

contracts-check:
	$(UV) run fraud-deploy contracts check

secret-scan:
	$(UV) run fraud-deploy secret-scan

db-up:
	docker compose up -d postgres
	docker compose ps postgres

db-down:
	docker compose down

db-status:
	docker compose ps postgres

db-test-create:
	docker compose exec -T postgres createdb -U fraud_dev fraud_detection_test

db-migrate:
	$(UV) run alembic upgrade head

db-downgrade:
	$(UV) run alembic downgrade base

db-reset:
	@test "$(CONFIRM)" = "reset" || (echo "Run: make db-reset CONFIRM=reset"; exit 1)
	docker compose down
	docker volume rm fraud_anomaly_detection_postgres_data
