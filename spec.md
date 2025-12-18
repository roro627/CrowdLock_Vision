# Human Lock-On Vision System

High‑level development specification for a real‑time, multi‑person human tracking and analytics system. The goal is to detect people in any video stream, compute precise target points (head center and body center), and expose the results to a modern web interface with real‑time overlays.

---

## 1. Goals & Scope

* Real‑time detection and tracking of multiple humans in arbitrary video streams.
* Compute and visualize **two distinct target points per person**:

  * **Head center** (between the eyes / center of the head region).
  * **Body center** (torso / belly area).
* Support **multiple camera perspectives**:

  * Front‑facing (security / webcam style).
  * Side views.
  * High‑angle views (e.g., ceiling cameras, mild aerial).
* Run on **live streams** and **recorded files**.
* Expose an HTTP/WebSocket API for a **web client** that renders the video with overlays in real time.
* Provide basic **crowd analytics**, including:

  * Total person count.
  * Spatial distribution of people.
  * Visual indication of the **most populated area** of the frame.

Out of scope:

* Any weapon / turret control.
* Face recognition or identity tracking beyond anonymous person IDs.

### Initial Project State

* This is a **greenfield / blank** project: there is no existing implementation.
* At the beginning, the repository is expected to contain only minimal scaffolding and a small set of **reference data** (e.g., sample clips or annotations) whose sole purpose is to verify the coherence and correctness of the future implementation.

---

## 2. Key Features

### 2.1 Person Detection & Pose Estimation

* Use a modern object detection / pose estimation model (e.g., YOLO‑based person detector + keypoints).
* Detect **all persons** in the frame (not just one).
* For each person:

  * Store bounding box.
  * Store pose keypoints when available (head, shoulders, hips, etc.).

### 2.2 Target Point Computation (Head & Body)

For every detected person:

* **Head center target**

  * Approximate using facial / head keypoints (eyes, nose, ears) when available.
  * Fallback: center of the upper region of the person bounding box.
  * Visual style (example): small circle with crosshair, color A.

* **Body center target**

  * Approximate using torso/hip keypoints (midpoint between left/right hips or between shoulders and hips).
  * Fallback: center of the full bounding box or lower‑middle region.
  * Visual style (example): different shape or color B (e.g., square marker, different stroke).

> Both targets are always computed and displayed **simultaneously**, never as an exclusive mode.

### 2.3 Multi‑Person Tracking

* Assign a stable **anonymous ID** to each person across frames (e.g., via ByteTrack / DeepSORT).
* Preserve track IDs as long as possible while persons stay in the scene.
* Expose tracking info to the web client for optional display (e.g., label each person with an ID).

### 2.4 Density Map & Most Populated Area

* Divide the frame into a configurable grid (e.g., 8×8 or 10×10).
* For every frame, map the **body center** of each person into a grid cell.
* Maintain per‑cell counts and compute:

  * Current **max density cell(s)**.
  * Optional smoothing over several frames to avoid flicker.
* Render a **heatmap overlay** on the video:

  * Semi‑transparent color map over cells.
  * Highlight the cell or region with the highest density.
* Expose density metrics through API (e.g., per‑cell counts, hottest region coordinates).

### 2.5 Video Source Abstraction

* Unified abstraction for multiple input types:

  * Local webcam(s).
  * Video files (e.g., MP4).
  * RTSP / HTTP streams (IP cameras, some drones, etc.).
* Configurable via a simple config file or environment variables.
* Hot‑switching between sources is a nice‑to‑have but not mandatory.

### 2.6 Performance & Real‑Time Constraints

* Target: near real‑time performance at common resolutions (e.g., 720p / 1080p) on a typical CPU‑only machine.
* Adjustable parameters:

  * Input frame size / downscaling.
  * Inference frequency (e.g., full detection every N frames, with tracking in between).
* Optional benchmarking/stats overlay:

  * FPS.
  * Inference time.
  * Number of detected persons.

---

## 3. System Architecture

The project is split into **clearly separated modules and folders** to keep development clean and maintainable.

### 3.1 High‑Level Components

1. **Core Vision Backend** (Python 3.12, or another suitable language for CV/ML):

   * Handles video capture, detection, pose estimation, tracking, and analytics.
   * Exposes a clean internal interface returning structured data per frame (persons, targets, density map, etc.).

2. **Web API Layer** (e.g., FastAPI / Node/Express):

   * Wraps the core vision backend and exposes:

     * REST endpoints for configuration, basic stats, and health checks.
     * WebSocket (or SSE) streams for real‑time frame metadata and video.

3. **Web Interface** (React + Tailwind CSS):

   * Modern, minimalist UI.
   * Displays the video feed with dynamic overlays.
   * Allows basic controls (select source, toggle overlays, etc.).

### 3.2 Suggested Folder Structure

```text
project-root/
  README.md
  backend/
    core/                    # Core CV / tracking / analytics logic
      __init__.py
      video_sources/         # Abstractions for webcams, files, RTSP streams
      detectors/             # Detection & pose models wrappers
      trackers/              # Multi-object tracking implementations
      analytics/             # Density map, stats, aggregation
      overlay/               # (Optional) server-side rendering helpers
      config/                # Config schemas & defaults
    api/                     # HTTP & WebSocket API layer
      main.py
      routes/
      schemas/
      services/
  web/
    package.json
    src/
      components/
      pages/
      hooks/
      lib/
      styles/                # Tailwind config & base styles
      api/                   # Client-side API / WebSocket utilities
  scripts/
    dev/                     # Dev helpers (start, lint, etc.)
    tools/                   # Utility scripts (e.g., dataset tools, profiling)
  config/
    app.example.env
    backend.config.example.yml
```

> Important: detectors, trackers, analytics, and web code **must live in separate subfolders**, not all in a single script.

---

## 4. Backend: Vision & Analytics

### 4.1 Detectors Module (`backend/core/detectors/`)

Responsibilities:

* Wrap model loading and inference.
* Provide a common interface, e.g.:

```python
class PersonDetector:
    def detect(self, frame) -> list[Detection]:
        """Return list of persons with bounding boxes and optional keypoints."""
```

Considerations:

* CPU-only backend.
* Configurable confidence thresholds and NMS settings.

### 4.2 Trackers Module (`backend/core/trackers/`)

Responsibilities:

* Maintain consistent IDs per person across frames.
* Accept raw detections and return tracked objects.
* Abstract underlying implementation (ByteTrack, DeepSORT, etc.).

### 4.3 Analytics Module (`backend/core/analytics/`)

Responsibilities:

* Compute per‑person target points (head center & body center).
* Build and update density maps over the frame grid.
* Optionally compute temporal stats (aggregated counts over time windows).

Outputs:

* For each frame, a structured summary:

```json
{
  "frame_id": 1234,
  "timestamp": 1680000000.123,
  "persons": [
    {
      "id": 1,
      "bbox": [x1, y1, x2, y2],
      "head_center": [xh, yh],
      "body_center": [xb, yb]
    },
    ...
  ],
  "density": {
    "grid_size": [10, 10],
    "cells": [...],
    "max_cell": [i, j]
  }
}
```

### 4.4 Video Sources Module (`backend/core/video_sources/`)

Responsibilities:

* Provide a unified API to consume frames:

```python
class VideoSource:
    def read(self) -> Frame | None: ...
    def close(self) -> None: ...
```

* Implementations:

  * `WebcamSource` (OpenCV index).
  * `FileSource` (video file path).
  * `RTSPSource` (stream URL).

### 4.5 API Layer (`backend/api/`)

Core endpoints:

* `GET /health` – health check.
* `GET /config` & `POST /config` – fetch/update basic settings.
* `GET /stats` – current aggregate stats (total persons, current FPS, etc.).
* `WS /stream/metadata` – WebSocket stream of per‑frame analytics data (JSON).
* `WS /stream/video` or alternative – video stream (MJPEG, WebRTC, or encoded frames + timestamps).

Security & deployment:

* Basic auth or token support (optional depending on target use case).
* Ready to be run in Docker.

---

## 5. Web Interface (React + Tailwind)

### 5.1 Design Principles

* Modern, minimal, and **clean** UI.
* Responsive layout (desktop‑first but usable on tablet).
* Light & dark mode if feasible (Tailwind dark variant).
* Focus on clarity of the overlays and stats.

### 5.2 Tech Stack

* **React** (SPA)
* **Tailwind CSS** for styling and layout.
* Optional extras:

  * Zustand or Redux for state management.
  * React Query or SWR for data fetching.

### 5.3 Core UI Elements

1. **Main Video Panel**

   * Displays live video from the backend.
   * Renders overlays on top:

     * Bounding boxes for each person.
     * **Head center marker** (shape/color A).
     * **Body center marker** (shape/color B).
     * Optional text label with person ID.
   * Overlays for density:

     * Semi‑transparent heatmap cells.
     * Highlighted region with highest population.

2. **Sidebar / Control Panel**

   * Source selection (dropdown: webcam, file, RTSP URL, etc.).
   * Toggles for visual layers:

     * Show/hide boxes.
     * Show/hide head markers.
     * Show/hide body markers.
     * Show/hide density heatmap.
   * Display key stats:

     * Current FPS.
     * Number of persons.
     * Coordinates / index of densest region.

3. **Footer / Status Bar**

   * Connection status to backend (WebSocket state).
   * Simple logs or last error message.

### 5.4 Web–Backend Communication

* WebSocket client in React responsible for:

  * Receiving frame metadata (persons, targets, density map).
  * Updating overlays in sync with the video.
* Video can be:

  * `<img>` tag bound to MJPEG stream.
  * `<video>` tag pulling HLS/WebRTC stream.
  * Canvas with raw frames if needed.

---

## 6. Configuration & Extensibility

### 6.1 Config Options

Provide configuration files or environment variables for at least:

* Model type and path.
* Inference performance settings (CPU-only).
* Input resolution and frame rate caps.
* Grid size for density analysis.
* Detection thresholds and tracking parameters.
* Video source selection / URLs.

Configuration should be centralized (e.g., `config/` folder with a main YAML/JSON file and environment overrides) so that the backend can be started in a well-defined, reproducible way.

### 6.2 Extensible Design

* Detectors, trackers, and analytics components must be **pluggable**:

  * Easy to add a new detector model by implementing a common interface and registering it.
  * Easy to switch trackers without touching the rest of the code.
  * Easy to plug in new analytics modules (e.g., dwell time per region, per-ID stats).
* The web client should allow future extension (new dashboards, charts, overlays) without breaking the existing layout.

---

## 7. Sample Test Videos for Vision

To validate and tune the vision pipeline (detection, target points, density map), the project must include a dedicated folder of example videos showing humans in a wide variety of situations.

### 7.1 Testdata Folder Structure

A dedicated `testdata/` folder must exist at the project root with at least the following minimal structure:

```text
testdata/
  videos/
    single_person_center.mp4
    two_people_left_right.mp4
    crowd_top_down.mp4
```

* `testdata/videos/` contains small, representative clips covering typical scenarios, for example:

  * Single person standing or walking near the center, front-facing.
  * Two persons separated in the frame (e.g., left vs right side).
  * A small crowd, possibly seen from above or at an angle, to validate density maps.
  * Additional clips with **mixed quality**: close-up faces, people far away, partial bodies, different lighting conditions, etc.

There is **no ground-truth JSON** that prescribes where targets "should" be. These videos are intended as:

* A shared visual reference for manual inspection.
* Inputs for generic automated consistency checks (e.g., bounding boxes inside frame, head above body, stable density maps), without any per-pixel or per-target annotation files.

### 7.2 Optional CLI for Debugging

The backend may provide a CLI entry point to run the vision pipeline on a given test video and dump diagnostic information (e.g., detections, head/body centers, density maps) to a log file or JSON output for debugging purposes, for example:

```bash
python -m backend.tools.run_on_video \
  --input testdata/videos/single_person_center.mp4 \
  --output tmp/single_person_center_output.json
```

However, this output is used only for **inspection and generic coherence checks**, not as a strict comparison to pre-defined ground-truth target positions.

## 8. Non‑Functional Requirements

* **Lightweight & optimized core**: aside from the web interface, all backend components (detectors, trackers, analytics, video pipeline, API) must be designed to be as **resource‑efficient** as possible for the target hardware, minimizing CPU usage, memory footprint, and unnecessary allocations while preserving real‑time behavior.
* **Code clarity**: small, focused modules; no monolithic scripts.
* **Logging**: structured logs for debugging performance and detection quality.
* **Testing**: unit tests for analytics logic (e.g., density grid, target point computation).
* **Portability & containerization**: should run on Linux and Windows; the backend **must have first-class Docker support** (Dockerfile + docker-compose configuration for the core services). The Dockerfile must be **optimized** (small base image, multi-stage builds when relevant, minimal runtime dependencies, efficient layer ordering) to reduce image size, build time, and runtime overhead.

---

## 9. Strict Definition of Done

The project must **not** be considered finished until all of the following conditions are met:

* All documented commands to install, build, run, and test the project execute **without errors or warnings**.
* **All tests pass** (green status) in the local environment and in the CI pipeline.
* No dependencies are flagged as **deprecated** or severely outdated, and the stack is kept as current as reasonably possible.
* **All features and modules described in this document are fully implemented**: no placeholder code, no stubbed functions, no `TODO` markers left in the codebase.
* The backend Docker setup is fully functional: `docker compose build` and `docker compose up` (or equivalent documented commands) complete **without errors or warnings**, and the system runs correctly inside containers.

This document serves as a consolidated development README describing the expected capabilities, architecture, and structure of the Human Lock-On Vision System without prescribing a specific implementation order or timeline.
