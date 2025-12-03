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

.PHONY: help setup install-backend install-frontend backend frontend dev dev-local

help:
	@echo "Available targets:"
	@echo "  setup             Install backend and frontend dependencies"
	@echo "  install-backend   Install Python dependencies for the backend"
	@echo "  install-frontend  Install Node dependencies for the web app"
	@echo "  backend           Start the FastAPI backend with uvicorn"
	@echo "  frontend          Start the Vite dev server"
	@echo "  dev               Start backend (docker by default) + frontend via start_stack.py"
	@echo "  dev-local         Run dev target with BACKEND_MODE=local"

setup: install-backend install-frontend

install-backend:
	$(PYTHON) -m pip install -r backend/requirements.txt

install-frontend:
	cd web && $(NPM) install

backend:
	PYTHONPATH=. $(UVICORN) $(BACKEND_APP) --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload

frontend:
	cd web && $(NPM) run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

dev:
	$(PYTHON) scripts/dev/start_stack.py --backend-mode $(BACKEND_MODE) --compose-command "$(COMPOSE_CMD)" --backend-host $(BACKEND_HOST) --backend-port $(BACKEND_PORT) --frontend-host $(FRONTEND_HOST) --frontend-port $(FRONTEND_PORT)

dev-local:
	$(MAKE) dev BACKEND_MODE=local
