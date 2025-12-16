from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from backend.core.analytics.targets import compute_targets
from backend.core.types import BBox, Detection, Point, TrackedPerson


def iou(boxA: BBox, boxB: BBox) -> float:
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
    id: int
    bbox: BBox
    head_center: Point
    body_center: Point
    confidence: float
    missed: int = 0


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, Track] = {}
        self._id_iter = itertools.count(1)

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        # Fast paths to avoid building matrices when possible.
        if not self.tracks:
            for det in detections:
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

        if not detections:
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

        assigned_tracks = set()
        assigned_dets = set()

        # Build IOU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=float)
        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = iou(self.tracks[tid].bbox, det.bbox)

        # Greedy assignment by IOU
        while True:
            if iou_matrix.size == 0:
                break
            ti, di = divmod(iou_matrix.argmax(), iou_matrix.shape[1])
            if iou_matrix[ti, di] < self.iou_threshold:
                break
            tid = track_ids[ti]
            track = self.tracks[tid]
            det = detections[di]
            head, body = compute_targets(det)
            self.tracks[tid] = Track(
                id=tid,
                bbox=det.bbox,
                head_center=head,
                body_center=body,
                confidence=det.confidence,
                missed=0,
            )
            assigned_tracks.add(tid)
            assigned_dets.add(di)
            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        # Add new tracks
        for di, det in enumerate(detections):
            if di in assigned_dets:
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

        # Age unmatched tracks
        to_delete = []
        for tid, track in self.tracks.items():
            if tid in assigned_tracks:
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
