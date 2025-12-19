"""Export the default YOLO model to ONNX (CPU-only).

This script is intentionally simple and print-oriented.
"""

from __future__ import annotations

import os


def main() -> int:
    """Run an ONNX export for the default model."""

    # Keep the project CPU-only: do not let Ultralytics auto-install GPU runtimes
    # (e.g. onnxruntime-gpu) as part of export.
    os.environ.setdefault("ULTRALYTICS_AUTOUPDATE", "0")

    from ultralytics import YOLO

    try:
        print("Loading model...")
        model_name = os.getenv("CLV_MODEL_NAME", "yolo11l.pt")
        model = YOLO(model_name)
        print("Exporting to ONNX...")
        model.export(format="onnx", device="cpu")
        print("Success!")
        return 0
    except Exception as e:
        print(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
