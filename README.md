# CrowdLock Vision

Real-time multi-person detection, tracking, and analytics with head/body lock-on targets and a React dashboard.

## Project Layout
```
backend/    FastAPI service + vision pipeline (YOLOv8 pose + tracker + analytics)
web/        React + Vite + Tailwind dashboard
scripts/    Dev helpers
config/     Example env and YAML config
testdata/   Sample videos for validation
```

## Quickstart (local)
1. Create and activate a Python env (example: `python3 -m venv .venv && source .venv/bin/activate`).
2. Install backend deps (CPU torch):
   ```bash
   PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install -r backend/requirements.txt
   ```
3. Start API (auto-loads config/backend.config.yml if present):
   ```bash
   export PYTHONPATH=.
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Frontend: `cd web && npm install && npm run dev -- --host` then open http://localhost:5173.
   - Dashboard shows live overlays and lets you change source (webcam/file/rtsp), model name, and confidence via the **Video Input** card.

## Config
- Copy `config/backend.config.example.yml` to `config/backend.config.yml` and adjust:
  - `video_source`: `webcam|file|rtsp`
  - `video_path` / `rtsp_url`
  - `model_name`, `confidence`, `grid_size`, `smoothing`
- Env vars override (prefix `CLV_`, see `config/app.example.env`).

## API
- `GET /health` – service ok
- `GET /config`, `POST /config` – view/update runtime config
- `GET /stats` – aggregate stats
- `GET /stream/video` – MJPEG stream with overlays
- `WS /stream/metadata` – per-frame JSON (persons, targets, density, fps)

## CLI
Process a video and dump JSON:
```bash
python -m backend.tools.run_on_video --input testdata/videos/855564-hd_1920_1080_24fps.mp4 --output tmp/out.json --max-frames 200
```

## Tests
- Backend unit tests: `pytest` (uses lightweight analytics logic; no heavy model needed).

## Docker
- Build images: `docker compose build`
- Run stack: `docker compose up`

## Definition of Done
- Full pipeline implemented: detection (YOLOv8 pose), tracking, target computation, density map, MJPEG + metadata streams.
- React dashboard renders overlays and controls.
- Tests provided for target computation and density smoothing.
