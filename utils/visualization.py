import numpy as np
import cv2
from collections import defaultdict

# Palet warna konsisten per track_id (random seed tetap, sama dengan test_run.py)
_rng_color = np.random.default_rng(42)
_id_colors: dict = {}


def get_color(tid: int) -> tuple:
    """Warna RGB konsisten per track ID (reuse dari test_run.py line 788-792)."""
    if tid not in _id_colors:
        _id_colors[tid] = tuple(_rng_color.integers(80, 220, size=3).tolist())
    return _id_colors[tid]


def draw_detections(frame: np.ndarray, boxes, confs, person_class: int = 0) -> np.ndarray:
    """
    Gambar bounding box deteksi pada frame (tanpa tracking ID).
    Warna: cyan (#00D4FF) untuk semua deteksi.
    """
    annotated = frame.copy()
    color = (0, 212, 255)  # BGR cyan

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"person {conf:.2f}"
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - lsz[1] - 6),
                      (x1 + lsz[0] + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated


def draw_tracking_frame(
    frame: np.ndarray,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    confs: np.ndarray,
    track_history: defaultdict,
    frame_idx: int,
) -> np.ndarray:
    """
    Gambar frame tracking dengan bounding box, label ID, dan trajectory.
    Reuse logika dari test_run.py lines 806-857.
    """
    annotated = frame.copy()

    for box, tid, conf in zip(boxes, track_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        track_history[int(tid)].append((cx, cy))

        color = get_color(int(tid))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label: Track ID + confidence
        label = f"ID:{tid} | {conf:.2f}"
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - lsz[1] - 6),
                      (x1 + lsz[0] + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Trajectory lines
        hist = track_history[int(tid)]
        for i in range(1, len(hist)):
            alpha_val = i / len(hist)
            cv2.line(annotated, hist[i - 1], hist[i],
                     color, max(1, int(alpha_val * 3)))

    # Info overlay
    cv2.putText(annotated, f"Frame {frame_idx} | ByteTrack | YOLOv11",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return annotated
