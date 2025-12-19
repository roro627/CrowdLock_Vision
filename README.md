<div align="center">

# 🎯 CrowdLock Vision

**Real-time multi-person detection, tracking & analytics**  
*CPU-optimized vision pipeline with frame-perfect overlay synchronization*

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev/)

[Features](#-features) • [Quick Start](#-quick-start) • [API](#-api-reference) • [Troubleshooting](#-troubleshooting)

</div>

---

## ✨ Features

- 🎥 **Multi-source support**: Webcam, video files, or RTSP streams
- 🧠 **YOLO11 inference**: CPU-optimized detection/pose with configurable stride
- 🎯 **Stable tracking**: Persistent IDs across frames with head/body center points
- 📊 **Density heatmaps**: Grid-based crowd analytics with smoothing
- ⚡ **Low-latency streaming**: MJPEG + WebSocket with frame-ID synchronization
- 🎨 **Live overlays**: React dashboard with perfectly aligned bounding boxes & metrics
- 🔧 **Runtime reconfiguration**: Change models, sources, and presets without restart
- 🐳 **Docker ready**: One-command deployment with docker-compose

---

## 🚀 Quick Start

### The fastest way to get started

**Requirements:** Python 3.12, Node.js + npm, and Docker (recommended for the backend).

```bash
# 1. Check your environment
make doctor

# 2. Launch backend + frontend (uses Docker by default)
make dev
```

**Then open:**

- 🌐 **Web UI**: <http://localhost:5173>
- 💚 **Backend health**: <http://localhost:8000/health>

> **💡 Tip**: No webcam? See [Demo Mode](#demo-mode-no-webcam-needed) below.

---

## 📂 Project Structure

```.
📁 CrowdLock_Vision/
├── 🐍 backend/         # FastAPI + vision pipeline (Ultralytics YOLO11)
│   ├── api/           # REST routes & WebSocket endpoints
│   ├── core/          # Detectors, trackers, analytics, video sources
│   └── tools/         # CLI utilities (bench, run-on-video)
├── ⚛️  web/            # React + TypeScript dashboard
│   ├── src/           # Components, hooks, overlays
│   └── public/        # Static assets
├── 🔧 config/          # YAML configs & env examples
├── 🐳 docker-compose.yml
├── 📊 testdata/videos/ # Sample clips for testing
└── 🛠️  scripts/        # Dev helpers (doctor, start_stack)
```

---

## 🔧 Manual Setup (Advanced)

1. Create and activate a Python env (example: `python3.12 -m venv .venv && source .venv/bin/activate`).
2. Install backend deps (CPU torch):

   ```bash
   PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install -r backend/requirements.txt
   ```

3. Start API (auto-loads config/backend.config.yml if present):

   ```bash
   export PYTHONPATH=.
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir backend --reload-dir config
   ```

   PowerShell equivalent:

   ```powershell
   $env:PYTHONPATH = "."
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir backend --reload-dir config
   ```

4. Frontend: `cd web && npm install && npm run dev -- --host` then open <http://localhost:5173>.
   - Dashboard shows live overlays and lets you change source (webcam/file/rtsp), model name, and confidence via the **Video Input** card.

### Development Stack Options

| Command | Backend | Use Case |
|---------|---------|----------|
| `make dev` | Docker | **Recommended** – isolated, reproducible |
| `make dev-local` | Local uvicorn | Fast iteration, debugging |
| `python scripts/dev/start_stack.py` | Configurable | Custom flags |

**See all targets**: `make help`

### Demo Mode (No Webcam Needed)

**Perfect for first-time users or CI/testing:**

1. Edit `config/backend.config.yml`:

   ```yaml
   video_source: file
   video_path: testdata/videos/855564-hd_1920_1080_24fps.mp4
   ```

2. Run `make dev`
3. Open <http://localhost:5173> and watch the pipeline process a pre-recorded crowd scene

> This validates the full stack (detection → tracking → analytics → streaming → overlays) without hardware dependencies.

---

---

## ⚙️ Configuration

### Config File (Recommended)

1. Copy the example: `cp config/backend.config.example.yml config/backend.config.yml`
2. Adjust key settings:

| Setting | Options | Description |
|---------|---------|-------------|
| `video_source` | `webcam`, `file`, `rtsp` | Input source type |
| `model_name` | `yolo11l.pt`, `yolo11l-pose.pt` | YOLO model variant |
| `confidence` | `0.0` – `1.0` | Detection threshold |
| `grid_size` | `10x10`, `16x16`, etc. | Density heatmap resolution |
| `inference_width` | `640`, `320`, etc. | Detector input size (↓ = faster) |
| `jpeg_quality` | `10` – `100` | Stream quality vs bandwidth |

### Environment Variables (Override)

Prefix any config key with `CLV_` (see `config/app.example.env`):

```bash
export CLV_VIDEO_SOURCE=webcam
export CLV_CONFIDENCE=0.4
```

---

## 📡 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/config` | GET / POST | View or update runtime configuration |
| `/stats` | GET | Aggregate analytics (people count, FPS, density) |
| `/stream/video` | GET | MJPEG stream with `X-Frame-Id` headers |

### WebSocket

| Endpoint | Protocol | Payload |
|----------|----------|---------|
| `/stream/metadata` | WS | Per-frame JSON: `{frame_id, timestamp, persons[], density{}, fps, stream_fps}` |

**Interactive docs**: <http://localhost:8000/docs> (Swagger UI)

---

## 🖥️ CLI Tools

### Process Video to JSON

```bash
python -m backend.tools.run_on_video \
  --input testdata/videos/855564-hd_1920_1080_24fps.mp4 \
  --output results.json \
  --max-frames 200
```

**Output**: Frame-by-frame analytics (persons, bboxes, density) in JSON format.

### Benchmark Pipeline Performance

**Identify bottlenecks** across decode → detect → overlay → encode stages:

```bash
# Test multiple presets
python -m backend.tools.bench_video \
  --input testdata/videos \
  --preset equilibre --preset fps_max --preset qualite

# Quick profiling
python -m backend.tools.bench_video \
  --input testdata/videos \
  --max-frames 150 --warmup-frames 20
```

**Output**: `benchmark_video_results.json` with per-stage latency percentiles (p50, p95, p99).

---

## 🧪 Testing

```bash
# Backend (no GPU required)
pytest -q

# Frontend
cd web && npm test
```

**Coverage**: Backend tests use lightweight mocks—no heavy YOLO inference needed.

---

## 🩺 Troubleshooting

| Symptom | Solution |
|---------|----------|
| 🖥️ **Web UI is black** | Ensure backend is running: <http://localhost:8000/health> |
| 📷 **Webcam not detected** | Switch to [Demo Mode](#demo-mode-no-webcam-needed) to validate pipeline |
| 🚪 **Port already in use** | Change ports in `Makefile` or stop conflicting process |
| 🐳 **Docker build fails** | Ensure Docker daemon is running: `docker info` |
| ⚠️ **Missing dependencies** | Run `make doctor` to check environment |

**Still stuck?** Run `make doctor` for a full environment audit.

---

## 🐳 Docker Deployment

```bash
# Build images
docker compose build

# Run stack (backend + web)
docker compose up
```

**Note**: This project is CPU-optimized by default. GPU acceleration requires custom `requirements.txt` changes.

### CPU-Only Design

Ultralytics auto-downloads models on first run. We disable GPU runtime auto-install via `ULTRALYTICS_AUTOUPDATE=0` to keep CPU torch wheels.

---

## 📦 Models & Assets

- **YOLO weights**: Auto-downloaded by Ultralytics on first inference (not versioned in repo)
- **Supported models**: `yolo11l.pt`, `yolo11l-pose.pt`, `yolo11s.pt`, etc.
- **Storage**: Models cached in `~/.cache/ultralytics/` (or `TORCH_HOME`)

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

<div align="center">

[⬆ Back to Top](#-crowdlock-vision)

</div>
