# CrowdLock Vision - Developer Guide

## Big picture (data flow)

- Backend targets **Python 3.12** and is a **CPU-only** real-time pipeline:
  `VideoSource` → `VisionPipeline` → REST/WS + MJPEG.
- Core loop lives in `backend/api/services/engine.py` (`VideoEngine`):
  - capture thread reads frames (`backend/core/video_sources/`)
  - process thread runs `VisionPipeline.process()` and optionally draws overlays
  - encode thread JPEG-encodes frames and drops old frames to keep latency low
- `VisionPipeline` (`backend/core/analytics/pipeline.py`) does:
  - detector (`backend/core/detectors/yolo.py`, Ultralytics YOLO) → detections
  - tracker (`backend/core/trackers/simple_tracker.py`) → `TrackedPerson` with stable IDs + head/body centers
  - density (`backend/core/analytics/density.py`) → grid heatmap + `max_cell`

## Project structure & module organization

- Authoritative requirements live in `spec.md`. Sample clips used for validation live in `testdata/videos/`.
- Target layout: `backend/` (core CV, trackers, analytics, API), `web/` (React UI), `scripts/` (dev/build helpers),
  `config/` (env examples). Keep detectors, trackers, analytics, and video sources in separate subfolders.
- Keep new assets small; large or proprietary footage should stay out of the repo unless explicitly cleared.

## API + web integration

- FastAPI app: `backend/api/main.py`.
- Streaming endpoints used by the web UI:
  - `GET /stream/video` (MJPEG) and `WS /stream/metadata` (per-frame JSON) in `backend/api/routes/stream.py`.
- Web app (`web/`) uses `VITE_API_BASE` (see `web/src/api/client.ts`) and renders overlays client-side
  (`web/src/components/VideoOverlay.tsx`).

## Developement environment

You has access to external tools and should use them when it improves correctness and relevance:

- **Context7**: fetch up-to-date library documentation and code examples (preferred when APIs may have changed).
- **Playwright**: drive a real browser to validate UI behavior, reproduce issues, or inspect pages.
- **web_search**: find general information and examples when the answer is not in the repo (useful for quick
  references and edge cases).

## Development rules

1. **DRY (Don't Repeat Yourself)**: avoid duplicating logic; extract shared helpers when reuse is real.
2. **KISS (Keep It Simple)**: prefer simple, explicit code over “clever” abstractions; most functions should be
  short and focused.
3. **Language**: all identifiers, comments, and documentation must be written in English.
4. **Docstrings**: every function must have a docstring describing what it does, its input parameters, and its
  return values (use the language-appropriate docstring style; for Python, prefer standard triple-quoted
  docstrings with type hints in the signature).
5. **Dependencies**: prefer widely used, well-maintained packages.
6. **Before adding a dependency**: check the official documentation and the date of the latest release; prefer
  libraries with clear maintenance and licensing.
7. **If a dependency is niche/risky**: justify the choice in this document and wrap it behind a small adapter
  module so it can be replaced later.
8. **No magic numbers**: avoid scattering hard-coded constants; name and centralize them.
9. **Centralize configuration**: behavior knobs must live in the project's configuration layer (for this repo:
  YAML defaults + `CLV_` environment overrides in `backend/core/config/settings.py` and `config/*.yml`).
10. **Anti-regression**: when fixing a bug, add or update a test that would have caught it.

## Config & runtime behavior

- Backend settings are **YAML defaults + env overrides** (env prefix `CLV_`), implemented in
  `backend/core/config/settings.py`.
  - Config file path is `CLV_CONFIG` (defaults to `config/backend.config.yml`).
  - `POST /config` triggers `backend/api/services/state.reload_settings()` which recreates the engine.
- Config: `.env` for secrets (never commit); YAML/JSON configs live in `config/` with `.example` templates.
- Do not hardcode credentials or stream URLs; load via environment and document in `config/*.example`.

## Performance knobs (tune before changing code)

- `inference_stride`: detector runs every N frames; skipped frames reuse last tracks.
- `inference_width`: passed as Ultralytics `imgsz` when it reduces work.
- `output_width` + `jpeg_quality`: reduces MJPEG encode cost without changing metadata coordinates.
- Optional CPU tuning: `CLV_TORCH_THREADS`, `CLV_TORCH_INTEROP_THREADS` (see `YoloPersonDetector`).

## Build, test, and development commands

- Backend: run via `uvicorn backend.api.main:app` (module path is `backend.api.main`; older notes may refer to
  `python -m backend.api.main`). Prefer `scripts/dev/start_backend.sh` for env setup.
- Frontend: from `web/`, `npm install` then `npm run dev` for local UI; `npm run build` for production bundle.
- Docker: `docker compose up --build` launches backend + web.

## Developer workflows (most common)

- Run dev stack: `make dev` (Docker backend by default) or `make dev-local` (local uvicorn). See
  `scripts/dev/start_stack.py`.
- Backend only: `uvicorn backend.api.main:app --reload --reload-dir backend --reload-dir config`
  (ensure `PYTHONPATH=.`).
- Web: `cd web && npm install && npm run dev -- --host`.
- Bench/CLI:
  - `python -m backend.tools.bench_video --input testdata/videos ...` (writes `benchmark_video_results.json`).
  - `python -m backend.tools.run_on_video --input ... --output ...`.

## Coding style & conventions

- Python: 4-space indent, type hints required, `black` + `ruff` for formatting/linting; snake_case modules,
  PascalCase classes, lower_snake functions/vars.
- JS/TS (web): `prettier` + `eslint` with TypeScript strict; React components PascalCase, hooks `useThing`,
  files kebab-case or component name.
- Python formatting/lint: Black + Ruff (`pyproject.toml`, line length 100). Keep type hints.
- Backend uses **Pydantic v2** with `pydantic-settings` (see `backend/requirements.txt`).
  Prefer keeping settings centralized in `backend/core/config/settings.py` and avoid ad-hoc env parsing.
- Docker images set `ULTRALYTICS_AUTOUPDATE=0` and use CPU Torch wheels (see `Dockerfile` + `backend/requirements.txt`).

## Testing guidelines

- Backend: `pytest` with >80% coverage target for analytics math; fixture videos come from `testdata/videos/`—keep
  tests deterministic (cap FPS, fixed seeds).
- Web: `npm run test` (Vitest/RTL) for components; snapshot overlays sparingly.
- Name tests `test_*.py` and `*.spec.tsx`; include at least one regression test per bug fix.
- Python tests: `pytest` (many tests use fakes/mocks so they don’t require heavy model inference).
- Web tests: `cd web && npm test`.
- **All code must be fully covered by tests, and all tests must pass without cheating, when running:**  
  `C:/Users/romai/AppData/Local/Programs/Python/Python312/python.exe -m pytest --cov=backend --cov-report=term-missing`

## Commit & pull request guidelines

- Repo history may be minimal (e.g., only an initial commit); adopt Conventional Commits
  (e.g., `feat: add density grid`, `fix: handle empty frame`).
- PRs: concise description, linked issue, testing notes (`pytest`, `npm test`, `docker compose up` if relevant),
  and screenshots/GIFs for UI or overlay changes.
- Keep PRs small and focused; include docs updates when adding commands or configs.

## Security & configuration tips

- Avoid committing large raw videos; reference paths under `testdata/videos/` or provide download instructions.
- Enable logging but redact PII; when exporting analytics, keep outputs anonymous (IDs only).
