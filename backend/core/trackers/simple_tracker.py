from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from backend.core.analytics.targets import compute_targets
from backend.core.types import BBox, Detection, Point, TrackedPerson

IOU_SUPPRESS_VALUE = -1.0


def iou(boxA: BBox, boxB: BBox) -> float:
    """Compute the intersection-over-union (IoU) of two axis-aligned boxes."""

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / float(union)


@dataclass
class Track:
    """Internal tracker state for one person."""

    id: int
    bbox: BBox
    head_center: Point
    body_center: Point
    confidence: float
    missed: int = 0


class SimpleTracker:
    """A lightweight IoU-based multi-object tracker.

    This tracker assigns detections to existing tracks greedily by IoU and keeps
    stable IDs for a short period of missed detections.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, Track] = {}
        self._id_iter = itertools.count(1)

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        """Update tracks from detector outputs and return current tracked persons."""

        detections_list = detections if isinstance(detections, list) else list(detections)

        # Fast paths to avoid building matrices when possible.
        if not self.tracks:
            for det in detections_list:
                new_id = next(self._id_iter)
                head, body = compute_targets(det)
                self.tracks[new_id] = Track(
                    id=new_id,
                    bbox=det.bbox,
                    head_center=head,
                    body_center=body,
                    confidence=det.confidence,
                    missed=0,
                )
            return [
                TrackedPerson(
                    id=track.id,
                    bbox=track.bbox,
                    head_center=track.head_center,
                    body_center=track.body_center,
                    confidence=track.confidence,
                )
                for track in self.tracks.values()
            ]

        if not detections_list:
            to_delete = []
            for tid, track in self.tracks.items():
                track.missed += 1
                if track.missed > self.max_missed:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return [
                TrackedPerson(
                    id=track.id,
                    bbox=track.bbox,
                    head_center=track.head_center,
                    body_center=track.body_center,
                    confidence=track.confidence,
                )
                for track in self.tracks.values()
            ]

        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]
        assigned_tracks = np.zeros(len(track_ids), dtype=bool)
        assigned_dets = np.zeros(len(detections_list), dtype=bool)

        track_boxes = np.array([track.bbox for track in track_list], dtype=np.float64)
        det_boxes = np.array([det.bbox for det in detections_list], dtype=np.float64)
        xA = np.maximum(track_boxes[:, None, 0], det_boxes[None, :, 0])
        yA = np.maximum(track_boxes[:, None, 1], det_boxes[None, :, 1])
        xB = np.minimum(track_boxes[:, None, 2], det_boxes[None, :, 2])
        yB = np.minimum(track_boxes[:, None, 3], det_boxes[None, :, 3])
        inter = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
        track_w = np.maximum(0.0, track_boxes[:, 2] - track_boxes[:, 0])
        track_h = np.maximum(0.0, track_boxes[:, 3] - track_boxes[:, 1])
        det_w = np.maximum(0.0, det_boxes[:, 2] - det_boxes[:, 0])
        det_h = np.maximum(0.0, det_boxes[:, 3] - det_boxes[:, 1])
        track_area = track_w * track_h
        det_area = det_w * det_h
        union = track_area[:, None] + det_area[None, :] - inter
        iou_matrix = np.where(union > 0.0, inter / union, 0.0)

        while True:
            if iou_matrix.size == 0:
                break
            ti, di = divmod(int(iou_matrix.argmax()), iou_matrix.shape[1])
            if iou_matrix[ti, di] < self.iou_threshold:
                break
            tid = track_ids[ti]
            track = track_list[ti]
            det = detections_list[di]
            head, body = compute_targets(det)
            track.bbox = det.bbox
            track.head_center = head
            track.body_center = body
            track.confidence = det.confidence
            track.missed = 0
            assigned_tracks[ti] = True
            assigned_dets[di] = True
            iou_matrix[ti, :] = IOU_SUPPRESS_VALUE
            iou_matrix[:, di] = IOU_SUPPRESS_VALUE

        for di, det in enumerate(detections_list):
            if assigned_dets[di]:
                continue
            new_id = next(self._id_iter)
            head, body = compute_targets(det)
            self.tracks[new_id] = Track(
                id=new_id,
                bbox=det.bbox,
                head_center=head,
                body_center=body,
                confidence=det.confidence,
                missed=0,
            )

        to_delete = []
        for ti, tid in enumerate(track_ids):
            if assigned_tracks[ti]:
                continue
            track = self.tracks.get(tid)
            if track is None:
                continue
            track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return [
            TrackedPerson(
                id=track.id,
                bbox=track.bbox,
                head_center=track.head_center,
                body_center=track.body_center,
                confidence=track.confidence,
            )
            for track in self.tracks.values()
        ]
