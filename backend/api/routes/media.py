"""Media helper endpoints.

These endpoints exist to support the web UI when selecting local demo assets.
They intentionally only expose files under `testdata/videos`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


class VideoFileInfo(BaseModel):
    """Represents a locally available demo video file.

    Attributes:
        name: Basename of the video file (e.g. "clip.mp4").
        path: Relative path usable in backend config (e.g. "testdata/videos/clip.mp4").
        thumbnail_url: Relative URL to fetch the JPEG thumbnail.
    """

    name: str
    path: str
    thumbnail_url: str


ROOT_DIR = Path(__file__).resolve().parents[3]
VIDEOS_DIR = ROOT_DIR / "testdata" / "videos"

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
THUMBNAIL_WIDTH_PX = 160
THUMBNAIL_JPEG_QUALITY = 70

router = APIRouter(prefix="/media", tags=["media"])


def _is_safe_basename(name: str) -> bool:
    """Return True if `name` is a safe filename (no path separators).

    Args:
        name: User-provided filename.

    Returns:
        True when the name is a basename (e.g. "a.mp4"), False otherwise.
    """

    if not name:
        return False
    if "/" in name or "\\" in name:
        return False
    return Path(name).name == name


def _iter_videos() -> list[VideoFileInfo]:
    """List available demo videos under testdata/videos.

    Returns:
        A list of VideoFileInfo objects sorted by filename.
    """

    if not VIDEOS_DIR.exists() or not VIDEOS_DIR.is_dir():
        return []

    items: list[VideoFileInfo] = []
    for p in sorted(VIDEOS_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
            continue
        name = p.name
        items.append(
            VideoFileInfo(
                name=name,
                path=f"testdata/videos/{name}",
                thumbnail_url=f"/media/videos/{quote(name)}/thumbnail.jpg",
            )
        )
    return items


@router.get("/videos", response_model=list[VideoFileInfo])
def list_videos() -> list[VideoFileInfo]:
    """List demo videos available on the backend host.

    Returns:
        List of available demo videos.
    """

    return _iter_videos()


@lru_cache(maxsize=256)
def _thumbnail_bytes_cache(name: str, mtime_ns: int) -> bytes:
    """Generate (and cache) a JPEG thumbnail for a video.

    Args:
        name: Video filename under `testdata/videos`.
        mtime_ns: File mtime in ns (part of the cache key).

    Returns:
        JPEG bytes.

    Raises:
        HTTPException: When the file cannot be opened/decoded.
    """

    video_path = VIDEOS_DIR / name
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise HTTPException(status_code=422, detail=f"Cannot read first frame: {name}")

    h, w = frame.shape[:2]
    if w > 0 and w != THUMBNAIL_WIDTH_PX:
        scale = THUMBNAIL_WIDTH_PX / float(w)
        new_h = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (THUMBNAIL_WIDTH_PX, new_h), interpolation=cv2.INTER_AREA)

    ok2, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(THUMBNAIL_JPEG_QUALITY)],
    )
    if not ok2:
        raise HTTPException(status_code=500, detail="Failed to encode thumbnail")

    return bytes(buf)


@router.get("/videos/{name}/thumbnail.jpg")
def video_thumbnail(name: str) -> Response:
    """Return a JPEG thumbnail (first frame) for a given demo video.

    Args:
        name: Basename of the video under `testdata/videos`.

    Returns:
        JPEG image response.
    """

    if not _is_safe_basename(name):
        raise HTTPException(status_code=400, detail="Invalid video name")

    path = VIDEOS_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    if path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=404, detail="Video not found")

    mtime_ns = path.stat().st_mtime_ns
    content = _thumbnail_bytes_cache(name, mtime_ns)
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )
