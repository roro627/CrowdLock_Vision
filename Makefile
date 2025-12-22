PYTHON ?= python
UVICORN ?= uvicorn
NPM ?= npm
BACKEND_APP ?= backend.api.main:app
BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173
BACKEND_MODE ?= docker
COMPOSE_CMD ?= docker compose

.PHONY: help setup install-backend install-frontend backend frontend dev dev-local doctor clean test

help:
	@echo "Available targets:"
	@echo "  setup             Install backend and frontend dependencies"
	@echo "  install-backend   Install Python dependencies for the backend"
	@echo "  install-frontend  Install Node dependencies for the web app"
	@echo "  backend           Start the FastAPI backend with uvicorn"
	@echo "  frontend          Start the Vite dev server"
	@echo "  dev               Start backend (docker by default) + frontend via start_stack.py"
	@echo "  dev-local         Run dev target with BACKEND_MODE=local"
	@echo "  doctor            Check environment + project sanity"
	@echo "  test              Run pytest with coverage (fails if <100%)"
	@echo "  clean             Remove temporary files/caches (incl. .venv and web/node_modules)"

setup: install-backend install-frontend

install-backend:
	$(PYTHON) -m pip install -r backend/requirements.txt

install-frontend:
	cd web && $(NPM) install

backend:
	PYTHONPATH=. $(UVICORN) $(BACKEND_APP) --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload --reload-dir backend --reload-dir config

frontend:
	cd web && $(NPM) run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

dev:
	$(PYTHON) scripts/dev/start_stack.py --backend-mode $(BACKEND_MODE) --compose-command "$(COMPOSE_CMD)" --backend-host $(BACKEND_HOST) --backend-port $(BACKEND_PORT) --frontend-host $(FRONTEND_HOST) --frontend-port $(FRONTEND_PORT)

dev-local:
	$(MAKE) dev BACKEND_MODE=local

doctor:
	$(PYTHON) scripts/dev/doctor.py

test:
	$(PYTHON) -m pytest --cov=backend --cov-report=term-missing --cov-fail-under=100

clean:
	@echo "Cleaning temporary files and folders..."
	@rm -rf \
		.pytest_cache \
		.ruff_cache \
		.mypy_cache \
		.hypothesis \
		.nox \
		.tox \
		.coverage \
		htmlcov \
		build \
		dist \
		*.egg-info \
		.eggs \
		.venv \
		web/dist \
		web/.vite \
		web/coverage \
		web/.coverage \
		web/node_modules
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@echo "Done."
