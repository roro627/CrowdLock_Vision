from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.types import BBox, Detection


@dataclass(frozen=True)
class RoiConfig:
    enabled: bool = False
    track_margin: float = 0.30
    entry_band: float = 0.08
    merge_iou: float = 0.20
    max_area_fraction: float = 0.70
    full_frame_every_n: int = 15
    force_full_frame_on_track_loss: float = 0.25
    detections_nms_iou: float = 0.50


@dataclass(frozen=True)
class PackedRoi:
    # ROI crop in original frame coords (float bbox, but typically integer-ish).
    roi: BBox
    # Top-left placement of the crop inside the mosaic.
    mosaic_x: int
    mosaic_y: int
    # Actual crop size (pixels).
    w: int
    h: int


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = _clamp(x1, 0.0, float(frame_w))
    x2 = _clamp(x2, 0.0, float(frame_w))
    y1 = _clamp(y1, 0.0, float(frame_h))
    y2 = _clamp(y2, 0.0, float(frame_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def expand_bbox(bbox: BBox, margin: float) -> BBox:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    mx = w * float(margin)
    my = h * float(margin)
    return (x1 - mx, y1 - my, x2 + mx, y2 + my)


def shift_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + float(dx), y1 + float(dy), x2 + float(dx), y2 + float(dy))


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    ua = bbox_area(a)
    ub = bbox_area(b)
    union = ua + ub - inter
    # Defensive: for well-formed boxes with inter>0, union should be >0.
    # Keep this guard but exclude from coverage (would require contrived NaN/inf inputs).
    if union <= 0.0:  # pragma: no cover
        return 0.0  # pragma: no cover
    return inter / union


def union_bbox(a: BBox, b: BBox) -> BBox:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def merge_rois(rois: list[BBox], iou_threshold: float) -> list[BBox]:
    if not rois:
        return []

    merged = [tuple(r) for r in rois]
    thr = float(iou_threshold)

    changed = True
    while changed:
        changed = False
        out: list[BBox] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = merged[i]
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if bbox_iou(cur, merged[j]) >= thr:
                    cur = union_bbox(cur, merged[j])
                    used[j] = True
                    changed = True
            out.append(cur)
        merged = out

    return merged


def build_rois_from_tracks(
    track_bboxes: list[BBox],
    frame_w: int,
    frame_h: int,
    *,
    track_margin: float,
    entry_band: float,
) -> list[BBox]:
    rois: list[BBox] = []

    for bbox in track_bboxes:
        rb = clamp_bbox(expand_bbox(bbox, track_margin), frame_w, frame_h)
        if bbox_area(rb) > 1.0:
            rois.append(rb)

    band = float(entry_band)
    if band > 0.0:
        bw = float(frame_w) * band
        bh = float(frame_h) * band
        if bw >= 1.0:
            rois.append((0.0, 0.0, bw, float(frame_h)))
            rois.append((float(frame_w) - bw, 0.0, float(frame_w), float(frame_h)))
        if bh >= 1.0:
            rois.append((0.0, float(frame_h) - bh, float(frame_w), float(frame_h)))

    # clamp everything
    return [clamp_bbox(r, frame_w, frame_h) for r in rois]


def nms_detections(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not detections:
        return []

    thr = float(iou_threshold)
    # sort by confidence desc
    dets = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
    kept: list[Detection] = []
    for det in dets:
        suppress = False
        for k in kept:
            if bbox_iou(det.bbox, k.bbox) >= thr:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept


def reproject_detection(det: Detection, dx: float, dy: float) -> Detection:
    x1, y1, x2, y2 = det.bbox
    kp = det.keypoints
    if kp is not None:
        kp = np.array(kp, copy=True)
        if kp.ndim >= 2 and kp.shape[1] >= 2:
            kp[:, 0] += float(dx)
            kp[:, 1] += float(dy)
    return Detection(
        bbox=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        confidence=det.confidence,
        keypoints=kp,
    )


def crop_bbox_to_int(bbox: BBox) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    xi1 = int(max(0.0, np.floor(x1)))
    yi1 = int(max(0.0, np.floor(y1)))
    xi2 = int(max(0.0, np.ceil(x2)))
    yi2 = int(max(0.0, np.ceil(y2)))
    if xi2 <= xi1:
        xi2 = xi1 + 1
    if yi2 <= yi1:
        yi2 = yi1 + 1
    return xi1, yi1, xi2, yi2


def pack_rois_grid(
    frame: np.ndarray,
    rois: list[BBox],
    *,
    max_cols: int = 2,
    pad: int = 2,
) -> tuple[np.ndarray, list[PackedRoi]]:
    """Pack multiple ROI crops into a single mosaic image.

    This reduces detector overhead by turning N ROI inferences into 1 inference on a smaller
    mosaic (when total ROI area is still much smaller than full frame).
    """

    if not rois:
        raise ValueError("rois must be non-empty")

    h, w = frame.shape[:2]
    cols = int(max(1, min(max_cols, len(rois))))
    rows = int((len(rois) + cols - 1) // cols)
    pad_i = int(max(0, pad))

    # Compute crop rects (clamped to frame) and their integer sizes.
    crops: list[tuple[BBox, int, int, int, int, int, int]] = []
    # (roi_bbox, x1,y1,x2,y2,cw,ch)
    for roi in rois:
        x1, y1, x2, y2 = crop_bbox_to_int(roi)
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        cw = int(x2 - x1)
        ch = int(y2 - y1)
        crops.append((roi, x1, y1, x2, y2, cw, ch))

    # Assign crops row-major.
    col_widths = [0] * cols
    row_heights = [0] * rows
    for idx, (_roi, _x1, _y1, _x2, _y2, cw, ch) in enumerate(crops):
        c = idx % cols
        r = idx // cols
        col_widths[c] = max(col_widths[c], cw)
        row_heights[r] = max(row_heights[r], ch)

    mosaic_w = int(sum(col_widths) + pad_i * (cols - 1))
    mosaic_h = int(sum(row_heights) + pad_i * (rows - 1))

    if frame.ndim == 3:
        mosaic = np.zeros((mosaic_h, mosaic_w, frame.shape[2]), dtype=frame.dtype)
    else:
        mosaic = np.zeros((mosaic_h, mosaic_w), dtype=frame.dtype)

    # Column/row offsets
    col_x = [0] * cols
    acc = 0
    for i, cw in enumerate(col_widths):
        col_x[i] = acc
        acc += int(cw) + (pad_i if i < cols - 1 else 0)
    row_y = [0] * rows
    acc = 0
    for i, rh in enumerate(row_heights):
        row_y[i] = acc
        acc += int(rh) + (pad_i if i < rows - 1 else 0)

    packed: list[PackedRoi] = []
    for idx, (roi, x1, y1, x2, y2, cw, ch) in enumerate(crops):
        c = idx % cols
        r = idx // cols
        mx = int(col_x[c])
        my = int(row_y[r])
        crop = frame[y1:y2, x1:x2]
        mosaic[my : my + ch, mx : mx + cw] = crop
        packed.append(PackedRoi(roi=roi, mosaic_x=mx, mosaic_y=my, w=cw, h=ch))

    return mosaic, packed


def pack_rois_best_grid(
    frame: np.ndarray,
    rois: list[BBox],
    *,
    max_cols_limit: int = 4,
    pad: int = 2,
) -> tuple[np.ndarray, list[PackedRoi]]:
    """Pack ROIs into a mosaic, choosing the grid width to minimize mosaic area.

    We keep the packing deterministic (row-major) and only search the number of columns.
    This can reduce mosaic area vs a fixed `max_cols=2`, which can improve FPS.
    """

    mosaic_w, mosaic_h, _area, best_cols = estimate_best_mosaic_area(
        frame_shape=frame.shape,
        rois=rois,
        max_cols_limit=max_cols_limit,
        pad=pad,
    )
    _ = (mosaic_w, mosaic_h)
    return pack_rois_grid(frame, rois, max_cols=best_cols, pad=int(max(0, pad)))


def estimate_best_mosaic_area(
    *,
    frame_shape: tuple[int, ...],
    rois: list[BBox],
    max_cols_limit: int = 4,
    pad: int = 2,
) -> tuple[int, int, int, int]:
    """Estimate the smallest possible mosaic area for a fixed row-major packing.

    Returns: (mosaic_w, mosaic_h, area, best_cols)
    """

    if not rois:
        raise ValueError("rois must be non-empty")

    h = int(frame_shape[0])
    w = int(frame_shape[1])
    pad_i = int(max(0, pad))

    crop_sizes: list[tuple[int, int]] = []
    for roi in rois:
        x1, y1, x2, y2 = crop_bbox_to_int(roi)
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        crop_sizes.append((int(x2 - x1), int(y2 - y1)))

    n = len(rois)
    max_cols_limit_i = int(max(1, max_cols_limit))
    best_cols = 1
    best_area: int | None = None
    best_w = 0
    best_h = 0

    for cols in range(1, min(max_cols_limit_i, n) + 1):
        rows = int((n + cols - 1) // cols)
        col_widths = [0] * cols
        row_heights = [0] * rows
        for idx, (cw, ch) in enumerate(crop_sizes):
            c = idx % cols
            r = idx // cols
            col_widths[c] = max(col_widths[c], cw)
            row_heights[r] = max(row_heights[r], ch)

        mosaic_w = int(sum(col_widths) + pad_i * (cols - 1))
        mosaic_h = int(sum(row_heights) + pad_i * (rows - 1))
        area = int(mosaic_w * mosaic_h)
        if best_area is None or area < best_area:
            best_area = area
            best_cols = cols
            best_w = mosaic_w
            best_h = mosaic_h

    return best_w, best_h, int(best_area or 0), best_cols


def split_and_reproject_mosaic_detections(
    detections: list[Detection],
    packed: list[PackedRoi],
) -> list[Detection]:
    """Assign each mosaic detection to a tile and reproject back to full-frame coords."""

    if not detections:
        return []
    if not packed:
        return []

    out: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        owner: PackedRoi | None = None
        for p in packed:
            if p.mosaic_x <= cx < (p.mosaic_x + p.w) and p.mosaic_y <= cy < (p.mosaic_y + p.h):
                owner = p
                break
        if owner is None:
            continue

        # dx/dy to map mosaic coords -> full frame coords
        rx1, ry1, _rx2, _ry2 = owner.roi
        dx = float(rx1) - float(owner.mosaic_x)
        dy = float(ry1) - float(owner.mosaic_y)
        out.append(reproject_detection(det, dx=dx, dy=dy))

    return out
