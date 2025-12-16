from __future__ import annotations

from backend.core.analytics.pipeline import VisionPipeline


class DetectorWithoutDetect:
    pass


def test_supports_imgsz_returns_false_on_introspection_error():
    pipeline = VisionPipeline(detector=DetectorWithoutDetect())
    assert pipeline._supports_imgsz() is False
