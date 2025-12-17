from __future__ import annotations

import numpy as np

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.roi import (
    RoiConfig,
    bbox_iou,
    build_rois_from_tracks,
    clamp_bbox,
    crop_bbox_to_int,
    estimate_best_mosaic_area,
    merge_rois,
    nms_detections,
    pack_rois_best_grid,
    pack_rois_grid,
    reproject_detection,
    shift_bbox,
    split_and_reproject_mosaic_detections,
)
from backend.core.types import Detection, TrackedPerson


class EchoTracker:
    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        return [
            TrackedPerson(
                id=1,
                bbox=d.bbox,
                head_center=(0.0, 0.0),
                body_center=(0.0, 0.0),
                confidence=d.confidence,
            )
            for d in detections
        ]


def test_build_rois_clamps_and_adds_entry_bands():
    rois = build_rois_from_tracks(
        track_bboxes=[(-10.0, -5.0, 30.0, 20.0)],
        frame_w=100,
        frame_h=50,
        track_margin=0.1,
        entry_band=0.1,
    )

    # Includes 1 track ROI + left/right + bottom bands.
    assert len(rois) >= 4

    for x1, y1, x2, y2 in rois:
        assert 0.0 <= x1 <= x2 <= 100.0
        assert 0.0 <= y1 <= y2 <= 50.0


def test_merge_rois_unions_overlapping():
    rois = [
        (0.0, 0.0, 10.0, 10.0),
        (5.0, 5.0, 15.0, 15.0),
    ]
    merged = merge_rois(rois, iou_threshold=0.05)
    assert len(merged) == 1
    assert merged[0] == (0.0, 0.0, 15.0, 15.0)


def test_merge_rois_empty_returns_empty():
    assert merge_rois([], iou_threshold=0.2) == []


def test_merge_rois_hits_used_j_skip_branch():
    # A merges with C (marking C used), leaving B to encounter used[j] in its inner loop.
    rois = [
        (0.0, 0.0, 10.0, 10.0),  # A
        (20.0, 20.0, 30.0, 30.0),  # B
        (1.0, 1.0, 9.0, 9.0),  # C overlaps A strongly
    ]
    out = merge_rois(rois, iou_threshold=0.5)
    assert len(out) == 2


def test_clamp_bbox_swaps_and_clamps():
    # inverted coordinates + out-of-bounds
    out = clamp_bbox((200.0, 60.0, -10.0, -5.0), frame_w=100, frame_h=50)
    assert out == (0.0, 0.0, 100.0, 50.0)


def test_crop_bbox_to_int_expands_degenerate_boxes():
    # xi2/yi2 must be forced to be > xi1/yi1
    x1, y1, x2, y2 = crop_bbox_to_int((5.0, 5.0, 5.0, 5.0))
    assert (x2 - x1) == 1
    assert (y2 - y1) == 1


def test_nms_detections_suppresses_overlaps():
    dets = [
        Detection(bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9),
        Detection(bbox=(1.0, 1.0, 9.0, 9.0), confidence=0.8),
        Detection(bbox=(50.0, 50.0, 60.0, 60.0), confidence=0.7),
    ]
    kept = nms_detections(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].confidence == 0.9


def test_bbox_iou_handles_non_overlapping_and_degenerate():
    assert bbox_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0
    # Degenerate (zero area) boxes => union == 0 path.
    assert bbox_iou((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)) == 0.0


def test_reproject_detection_shifts_bbox_and_keypoints():
    kp = np.array([[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]], dtype=float)
    det = Detection(bbox=(10.0, 20.0, 30.0, 40.0), confidence=0.9, keypoints=kp)
    out = reproject_detection(det, dx=5.0, dy=-2.0)

    assert out.bbox == (15.0, 18.0, 35.0, 38.0)
    assert out.keypoints is not None
    assert float(out.keypoints[0, 0]) == 6.0
    assert float(out.keypoints[0, 1]) == 0.0


def test_reproject_detection_keeps_none_keypoints_branch():
    det = Detection(bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.5, keypoints=None)
    out = reproject_detection(det, dx=1.0, dy=1.0)
    assert out.keypoints is None


def test_shift_bbox_shifts_coordinates():
    assert shift_bbox((1.0, 2.0, 3.0, 4.0), dx=5.0, dy=-1.0) == (6.0, 1.0, 8.0, 3.0)


def test_pack_rois_grid_packs_two_tiles_and_returns_packed_rois():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # Make each ROI crop distinct.
    frame[0:3, 0:4] = 10
    frame[6:10, 6:10] = 200

    rois = [(0.0, 0.0, 4.0, 3.0), (6.0, 6.0, 10.0, 10.0)]
    mosaic, packed = pack_rois_grid(frame, rois, max_cols=2, pad=1)

    assert len(packed) == 2
    # With 2 cols and pad=1: width=4 + 1 + 4 = 9, height=max(3,4)=4
    assert mosaic.shape[:2] == (4, 9)
    # First tile should be present at (0,0)
    assert int(mosaic[0, 0, 0]) == 10
    # Second tile should be present after padding
    assert int(mosaic[3, 8, 0]) == 200


def test_pack_rois_best_grid_chooses_smaller_area_than_fixed_two_cols():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    # Three crops where 1 column should be better than 2 columns:
    # - Two very tall crops + one tiny crop.
    rois = [
        (0.0, 0.0, 2.0, 20.0),
        (2.0, 0.0, 4.0, 20.0),
        (4.0, 0.0, 6.0, 2.0),
    ]
    mosaic2, _ = pack_rois_grid(frame, rois, max_cols=2, pad=2)
    mosaic_best, _ = pack_rois_best_grid(frame, rois, max_cols_limit=4, pad=2)
    area2 = int(mosaic2.shape[0] * mosaic2.shape[1])
    area_best = int(mosaic_best.shape[0] * mosaic_best.shape[1])
    assert area_best <= area2


def test_estimate_best_mosaic_area_matches_actual_best_grid_shape():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    rois = [
        (0.0, 0.0, 10.0, 10.0),
        (10.0, 0.0, 20.0, 10.0),
        (0.0, 10.0, 10.0, 20.0),
    ]
    mw, mh, area, _cols = estimate_best_mosaic_area(
        frame_shape=frame.shape, rois=rois, max_cols_limit=4, pad=2
    )
    mosaic, _ = pack_rois_best_grid(frame, rois, max_cols_limit=4, pad=2)
    assert (mh, mw) == mosaic.shape[:2]
    assert area == int(mosaic.shape[0] * mosaic.shape[1])


def test_split_and_reproject_mosaic_detections_assigns_owner_and_reprojects():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    rois = [(0.0, 0.0, 4.0, 3.0), (6.0, 6.0, 10.0, 10.0)]
    mosaic, packed = pack_rois_grid(frame, rois, max_cols=2, pad=1)
    _ = mosaic

    # Detection centered in first tile at mosaic coords ~ (2,1)
    det0 = Detection(bbox=(1.0, 0.5, 3.0, 1.5), confidence=0.9)
    # Detection centered in second tile at mosaic coords ~ (6,2)
    det1 = Detection(bbox=(5.0, 1.0, 7.0, 3.0), confidence=0.8)
    # Detection outside any tile should be dropped
    det2 = Detection(bbox=(100.0, 100.0, 110.0, 110.0), confidence=0.7)

    out = split_and_reproject_mosaic_detections([det0, det1, det2], packed)
    assert len(out) == 2
    assert out[0].bbox == det0.bbox
    # Second ROI is placed after first tile + pad => mosaic_x == 5
    # Original ROI top-left is (6,6), so dx=+1, dy=+6
    assert out[1].bbox == (6.0, 7.0, 8.0, 9.0)


class ShapeRecordingDetector:
    def __init__(self):
        self.shapes: list[tuple[int, int]] = []

    def detect(self, frame, imgsz=None):
        h, w = frame.shape[:2]
        self.shapes.append((h, w))
        return [
            Detection(
                bbox=(w * 0.4, h * 0.4, w * 0.6, h * 0.6),
                confidence=0.9,
                keypoints=None,
            )
        ]


class TwoDetectionsRecordingDetector:
    def __init__(self):
        self.shapes: list[tuple[int, int]] = []

    def detect(self, frame, imgsz=None):
        h, w = frame.shape[:2]
        self.shapes.append((h, w))
        return [
            Detection(bbox=(w * 0.10, h * 0.10, w * 0.20, h * 0.20), confidence=0.9),
            Detection(bbox=(w * 0.70, h * 0.70, w * 0.80, h * 0.80), confidence=0.8),
        ]


def test_pipeline_roi_uses_crop_after_initial_full_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = ShapeRecordingDetector()
    pipeline = VisionPipeline(
        detector=detector,
        tracker=EchoTracker(),
        roi_config=RoiConfig(
            enabled=True,
            track_margin=0.2,
            entry_band=0.0,
            merge_iou=0.2,
            max_area_fraction=0.9,
            full_frame_every_n=0,
            force_full_frame_on_track_loss=1.0,
        ),
    )

    pipeline.process(frame, inference_stride=1)
    pipeline.process(frame, inference_stride=1)

    assert detector.shapes[0] == (100, 100)
    # Second inference should run on a crop smaller than full frame.
    assert detector.shapes[1][0] < 100
    assert detector.shapes[1][1] < 100


def test_pipeline_roi_uses_mosaic_when_multiple_rois_present():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = TwoDetectionsRecordingDetector()
    pipeline = VisionPipeline(
        detector=detector,
        tracker=EchoTracker(),
        roi_config=RoiConfig(
            enabled=True,
            track_margin=0.0,
            entry_band=0.0,
            merge_iou=0.0,
            max_area_fraction=0.9,
            full_frame_every_n=0,
            force_full_frame_on_track_loss=1.0,
        ),
    )

    pipeline.process(frame, inference_stride=1)
    pipeline.process(frame, inference_stride=1)

    assert detector.shapes[0] == (100, 100)
    # Second inference should be on a packed mosaic (smaller than full frame).
    assert detector.shapes[1][0] < 100
    assert detector.shapes[1][1] < 100


def test_pipeline_roi_area_cap_forces_full_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = ShapeRecordingDetector()
    pipeline = VisionPipeline(
        detector=detector,
        tracker=EchoTracker(),
        roi_config=RoiConfig(
            enabled=True,
            track_margin=10.0,  # expands to full-frame after clamp
            entry_band=0.0,
            merge_iou=0.2,
            max_area_fraction=0.5,
            full_frame_every_n=0,
            force_full_frame_on_track_loss=1.0,
        ),
    )

    pipeline.process(frame, inference_stride=1)
    pipeline.process(frame, inference_stride=1)

    assert detector.shapes[0] == (100, 100)
    # ROI would cover the full frame -> should fallback to full frame.
    assert detector.shapes[1] == (100, 100)


class FlakyDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame, imgsz=None):
        self.calls += 1
        h, w = frame.shape[:2]
        if self.calls == 1:
            return [Detection(bbox=(w * 0.4, h * 0.4, w * 0.6, h * 0.6), confidence=0.9)]
        return []


def test_pipeline_roi_profile_includes_roi_stats_and_track_loss_trigger():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = FlakyDetector()
    pipeline = VisionPipeline(
        detector=detector,
        tracker=EchoTracker(),
        roi_config=RoiConfig(
            enabled=True,
            track_margin=0.2,
            entry_band=0.0,
            merge_iou=0.2,
            max_area_fraction=0.9,
            full_frame_every_n=0,
            force_full_frame_on_track_loss=0.25,
        ),
    )

    # First inference creates 1 track (full frame forced by empty last_persons).
    s1, _, t1 = pipeline.process_with_profile(frame, inference_stride=1)
    assert len(s1.persons) == 1
    assert "roi_used" in t1

    # Second inference returns no detections => track-loss triggers and profile captures it.
    s2, _, t2 = pipeline.process_with_profile(frame, inference_stride=1)
    assert len(s2.persons) == 0
    assert t2.get("track_loss_frac", 0.0) >= 0.99
