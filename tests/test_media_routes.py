from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import media


def test_media_list_videos_filters_and_sorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    (tmp_path / "c.MOV").write_bytes(b"x")
    (tmp_path / "a-folder").mkdir()

    monkeypatch.setattr(media, "VIDEOS_DIR", tmp_path)
    client = TestClient(app)
    res = client.get("/media/videos")
    assert res.status_code == 200
    data = res.json()
    assert [v["name"] for v in data] == ["a.mp4", "b.mp4", "c.MOV"]
    assert data[0]["path"] == "testdata/videos/a.mp4"
    assert data[0]["thumbnail_url"].endswith("/thumbnail.jpg")


def test_media_list_videos_when_folder_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(media, "VIDEOS_DIR", missing)
    client = TestClient(app)
    res = client.get("/media/videos")
    assert res.status_code == 200
    assert res.json() == []


def test_media_is_safe_basename_rejects_empty_string():
    assert media._is_safe_basename("") is False


def test_media_thumbnail_happy_path_no_resize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "ok.mp4").write_bytes(b"x")
    monkeypatch.setattr(media, "VIDEOS_DIR", tmp_path)
    media._thumbnail_bytes_cache.cache_clear()

    class FakeCap:
        def read(self):
            frame = np.zeros((90, media.THUMBNAIL_WIDTH_PX, 3), dtype=np.uint8)
            return True, frame

        def release(self):
            return None

    monkeypatch.setattr(media.cv2, "VideoCapture", lambda _: FakeCap())
    monkeypatch.setattr(media.cv2, "imencode", lambda *_args, **_kw: (True, b"JPEG"))

    client = TestClient(app)
    res = client.get("/media/videos/ok.mp4/thumbnail.jpg")
    assert res.status_code == 200
    assert res.headers.get("content-type", "").startswith("image/jpeg")
    assert res.content == b"JPEG"


def test_media_thumbnail_happy_path_with_resize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "resize.mp4").write_bytes(b"x")
    monkeypatch.setattr(media, "VIDEOS_DIR", tmp_path)
    media._thumbnail_bytes_cache.cache_clear()

    class FakeCap:
        def read(self):
            frame = np.zeros((100, 320, 3), dtype=np.uint8)
            return True, frame

        def release(self):
            return None

    monkeypatch.setattr(media.cv2, "VideoCapture", lambda _: FakeCap())
    monkeypatch.setattr(
        media.cv2,
        "resize",
        lambda frame, _size, interpolation=None: frame[:, : media.THUMBNAIL_WIDTH_PX, :],
    )
    monkeypatch.setattr(media.cv2, "imencode", lambda *_args, **_kw: (True, b"JPEG2"))

    client = TestClient(app)
    res = client.get("/media/videos/resize.mp4/thumbnail.jpg")
    assert res.status_code == 200
    assert res.content == b"JPEG2"


def test_media_thumbnail_read_failure_returns_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "bad.mp4").write_bytes(b"x")
    monkeypatch.setattr(media, "VIDEOS_DIR", tmp_path)
    media._thumbnail_bytes_cache.cache_clear()

    class FakeCap:
        def read(self):
            return False, None

        def release(self):
            return None

    monkeypatch.setattr(media.cv2, "VideoCapture", lambda _: FakeCap())

    client = TestClient(app)
    res = client.get("/media/videos/bad.mp4/thumbnail.jpg")
    assert res.status_code == 422


def test_media_thumbnail_encode_failure_returns_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "enc.mp4").write_bytes(b"x")
    monkeypatch.setattr(media, "VIDEOS_DIR", tmp_path)
    media._thumbnail_bytes_cache.cache_clear()

    class FakeCap:
        def read(self):
            frame = np.zeros((10, media.THUMBNAIL_WIDTH_PX, 3), dtype=np.uint8)
            return True, frame

        def release(self):
            return None

    monkeypatch.setattr(media.cv2, "VideoCapture", lambda _: FakeCap())
    monkeypatch.setattr(media.cv2, "imencode", lambda *_args, **_kw: (False, None))

    client = TestClient(app)
    res = client.get("/media/videos/enc.mp4/thumbnail.jpg")
    assert res.status_code == 500


def test_media_thumbnail_rejects_backslash_and_encoded_slash():
    client = TestClient(app)

    # Encoded backslash is decoded into a single path segment, so the route matches and
    # our safety check should reject it.
    res1 = client.get("/media/videos/..%5Csecret/thumbnail.jpg")
    assert res1.status_code == 400

    # Encoded slash is treated as a path separator by the router, so the route doesn't match.
    res2 = client.get("/media/videos/..%2Fsecret/thumbnail.jpg")
    assert res2.status_code == 404


def test_media_thumbnail_404_on_missing_and_bad_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "note.txt").write_bytes(b"x")
    monkeypatch.setattr(media, "VIDEOS_DIR", tmp_path)
    media._thumbnail_bytes_cache.cache_clear()

    client = TestClient(app)
    assert client.get("/media/videos/missing.mp4/thumbnail.jpg").status_code == 404
    assert client.get("/media/videos/note.txt/thumbnail.jpg").status_code == 404
