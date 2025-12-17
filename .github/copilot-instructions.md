# Copilot instructions (CrowdLock Vision)

## Big picture (data flow)
- Backend is a **CPU-only** real-time pipeline: `VideoSource` → `VisionPipeline` → REST/WS + MJPEG.
- Core loop lives in `backend/api/services/engine.py` (`VideoEngine`):
  - capture thread reads frames (`backend/core/video_sources/`)
  - process thread runs `VisionPipeline.process()` and optionally draws overlays
  - encode thread JPEG-encodes frames and drops old frames to keep latency low
- `VisionPipeline` (`backend/core/analytics/pipeline.py`) does:
  - detector (`backend/core/detectors/yolo.py`, Ultralytics YOLO) → detections
  - tracker (`backend/core/trackers/simple_tracker.py`) → `TrackedPerson` with stable IDs + head/body centers
  - density (`backend/core/analytics/density.py`) → grid heatmap + `max_cell`

## API + web integration
- FastAPI app: `backend/api/main.py`.
- Streaming endpoints used by the web UI:
  - `GET /stream/video` (MJPEG) and `WS /stream/metadata` (per-frame JSON) in `backend/api/routes/stream.py`.
- Web app (`web/`) uses `VITE_API_BASE` (see `web/src/api/client.ts`) and renders overlays client-side (`web/src/components/VideoOverlay.tsx`).

## Config & runtime behavior
- Backend settings are **YAML defaults + env overrides** (env prefix `CLV_`), implemented in `backend/core/config/settings.py`.
  - Config file path is `CLV_CONFIG` (defaults to `config/backend.config.yml`).
  - `POST /config` triggers `backend/api/services/state.reload_settings()` which recreates the engine.
- Performance knobs (tune before changing code):
  - `inference_stride`: detector runs every N frames; skipped frames reuse last tracks.
  - `inference_width`: passed as Ultralytics `imgsz` when it reduces work.
  - `output_width` + `jpeg_quality`: reduces MJPEG encode cost without changing metadata coordinates.
  - Optional CPU tuning: `CLV_TORCH_THREADS`, `CLV_TORCH_INTEROP_THREADS` (see `YoloPersonDetector`).

## Developer workflows (most common)
- Run dev stack: `make dev` (Docker backend by default) or `make dev-local` (local uvicorn). See `scripts/dev/start_stack.py`.
- Backend only: `uvicorn backend.api.main:app --reload --reload-dir backend --reload-dir config` (ensure `PYTHONPATH=.`).
- Web: `cd web && npm install && npm run dev -- --host`.
- Bench/CLI:
  - `python -m backend.tools.bench_video --input testdata/videos ...` (writes `benchmark_video_results.json`).
  - `python -m backend.tools.run_on_video --input ... --output ...`.
- Tests:
  - Python: `pytest` (many tests use fakes/mocks so they don’t require heavy model inference).
  - Web: `cd web && npm test`.
- **All code must be fully covered by tests, and all tests must pass without cheating, when running:**  
  `C:/Users/romai/AppData/Local/Programs/Python/Python311/python.exe -m pytest --cov=backend --cov-report=term-missing`

## Project conventions
- Python formatting/lint: Black + Ruff (`pyproject.toml`, line length 100). Keep type hints.
- Don’t introduce `pydantic-settings` unless you intentionally migrate to Pydantic v2 (backend pins `pydantic==1.10.18`).
- Docker images set `ULTRALYTICS_AUTOUPDATE=0` and use CPU Torch wheels (see `Dockerfile` + `backend/requirements.txt`).
- Keep large/proprietary videos out of the repo; use `testdata/videos/` for fixtures.
