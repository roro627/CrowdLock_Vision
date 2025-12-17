"""Ultralytics YOLO detector integration.

This module intentionally keeps Torch as an optional runtime dependency: ONNX
exports can run without importing torch.
"""

from __future__ import annotations

import importlib
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
from ultralytics import YOLO

from backend.core.types import Detection


class YoloPersonDetector:
    """Person detector wrapper around Ultralytics YOLO.

    Supports both Torch `.pt` models and ONNX exports. The detector is CPU-only by
    default and can optionally be tuned via env vars (`CLV_TORCH_THREADS`,
    `CLV_TORCH_INTEROP_THREADS`).
    """

    _torch_threads_configured: bool = False

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf: float = 0.3,
        task: str | None = None,
    ):
        """Create a detector.

        Args:
            model_name: Model path/name understood by Ultralytics (e.g. `yolov8n.pt`,
                `yolov8n-pose.pt`, or an `.onnx` export).
            conf: Confidence threshold applied inside the Ultralytics predictor.
            task: Optional Ultralytics task override.
        """

        # Optional CPU tuning. This is intentionally env-driven so production
        # deployments can choose the best value per machine without code changes.
        # Defaults are left untouched unless explicitly configured.
        self._configure_torch_threads_from_env()

        self.model_name = model_name
        self.is_onnx = model_name.lower().endswith(".onnx")
        self.device: str = "cpu"
        # Avoid .to(device) on ONNX exports; Ultralytics raises TypeError
        self.model = YOLO(model_name, task=task)

        # Force torch models to CPU.
        if not self.is_onnx:
            try:
                self.model.to(self.device)
            except Exception:
                # Some backends may not support .to(); predict(device='cpu') still enforces CPU.
                pass
        self.conf = conf
        self._base_predict_kwargs = {
            "conf": self.conf,
            "verbose": False,
            # Ask Ultralytics to keep only the person class early (COCO class id 0).
            # This reduces NMS/post-processing overhead on CPU.
            "classes": [0],
            "device": self.device,
        }

        # Small CPU speed win (Conv+BN fusion) for torch models.
        if not self.is_onnx:
            try:
                self.model.fuse()
            except Exception:
                # Some Ultralytics versions/models may not support fuse(); ignore.
                pass

    @classmethod
    def _configure_torch_threads_from_env(cls) -> None:
        """Configure torch thread counts from environment variables (one-time)."""

        if cls._torch_threads_configured:
            return
        cls._torch_threads_configured = True

        threads_s = os.getenv("CLV_TORCH_THREADS")
        interop_s = os.getenv("CLV_TORCH_INTEROP_THREADS")
        if threads_s is None and interop_s is None:
            return

        try:
            torch = importlib.import_module("torch")

            if threads_s is not None and threads_s.strip():
                torch.set_num_threads(max(1, int(threads_s)))
            if interop_s is not None and interop_s.strip():
                torch.set_num_interop_threads(max(1, int(interop_s)))
        except Exception:
            # If torch isn't present or refuses changes, ignore.
            return

    def detect(self, frame: np.ndarray, imgsz: int | None = None) -> list[Detection]:
        """Run inference on a single frame and return person detections.

        Args:
            frame: Input image as a numpy array in OpenCV format.
            imgsz: Optional Ultralytics inference size. When provided, it is used
                only as a *downscale* hint to reduce CPU work (never to upscale).

        Returns:
            A list of `Detection` objects in full-frame pixel coordinates.
        """

        kwargs = self._base_predict_kwargs
        if imgsz:
            # Ultralytics expects `imgsz` to drive its internal letterboxing.
            kwargs = dict(kwargs)
            kwargs["imgsz"] = int(imgsz)

        # Ultralytics already uses no-grad internally in most cases, but being explicit
        # avoids surprises and keeps CPU execution lean.
        try:
            torch: Any = importlib.import_module("torch")
            infer_ctx = torch.inference_mode() if not self.is_onnx else nullcontext()
        except Exception:
            infer_ctx = nullcontext()

        with infer_ctx:
            results = self.model.predict(frame, **kwargs)

        if not results:
            return []

        # Single-frame inference => first result.
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # Prefer `boxes.data` to pull xyxy/conf in one tensor (less attribute churn).
        data = getattr(boxes, "data", None)
        if data is None:
            xyxy = getattr(boxes, "xyxy", None)
            confs = getattr(boxes, "conf", None)
            if xyxy is None or confs is None:
                return []

            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu()
            if hasattr(confs, "cpu"):
                confs = confs.cpu()

            xyxy_np = xyxy.numpy() if hasattr(xyxy, "numpy") else np.asarray(xyxy)
            confs_np = confs.numpy() if hasattr(confs, "numpy") else np.asarray(confs)
        else:
            if hasattr(data, "cpu"):
                data = data.cpu()
            data_np = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
            # Ultralytics Boxes.data = (x1,y1,x2,y2,conf,cls)
            if data_np.ndim != 2 or data_np.shape[1] < 5:
                return []
            xyxy_np = data_np[:, :4]
            confs_np = data_np[:, 4]

        kpts = getattr(result, "keypoints", None)
        kpts_np = None
        if kpts is not None and hasattr(kpts, "data") and kpts.data is not None:
            kd = kpts.data
            if hasattr(kd, "cpu"):
                kd = kd.cpu()
            kpts_np = kd.numpy() if hasattr(kd, "numpy") else np.asarray(kd)

        out: list[Detection] = []
        if kpts_np is None:
            # Iterate numpy rows directly to avoid large intermediate Python lists.
            for bbox, conf_v in zip(xyxy_np, confs_np, strict=False):
                out.append(
                    Detection(
                        bbox=(
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ),
                        confidence=float(conf_v),
                        keypoints=None,
                    )
                )
            return out

        # Pose model: attach per-detection keypoints.
        for i, (bbox, conf_v) in enumerate(zip(xyxy_np, confs_np, strict=False)):
            kp = kpts_np[i] if i < int(kpts_np.shape[0]) else None
            out.append(
                Detection(
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    confidence=float(conf_v),
                    keypoints=kp,
                )
            )
        return out
