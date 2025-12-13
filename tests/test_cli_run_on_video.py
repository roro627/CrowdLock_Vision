import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


def _make_dummy_video(path: Path, frames: int = 5, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, size)
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG video on this platform")
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def test_run_on_video_cli(tmp_path: Path):
    video_path = tmp_path / "dummy.avi"
    out_path = tmp_path / "out.json"
    _make_dummy_video(video_path)

    cap = cv2.VideoCapture(str(video_path))
    read_result = cap.read()
    cap.release()
    if not isinstance(read_result, tuple) or len(read_result) != 2:
        pytest.skip("OpenCV backend cannot read generated video on this platform")
    ok, frame = read_result
    if not ok or frame is None:
        pytest.skip("OpenCV backend cannot read generated video on this platform")

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".") + os.pathsep + str(Path(__file__).resolve().parents[1])

    cmd = [
        sys.executable,
        "-m",
        "backend.tools.run_on_video",
        "--input",
        str(video_path),
        "--output",
        str(out_path),
        "--max-frames",
        "2",
        "--mock",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    # file should contain at most 2 summaries
    import json

    data = json.loads(out_path.read_text())
    assert len(data) <= 2
