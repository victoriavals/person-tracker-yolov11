import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from collections import defaultdict
from typing import Callable
from ultralytics import YOLO

from utils.visualization import draw_tracking_frame
from utils.model import PERSON_CLASS


def get_video_info(video_path: str) -> dict:
    """Ambil metadata video (lebar, tinggi, FPS, total frame)."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def track_video(
    model: YOLO,
    input_path: str,
    conf: float = 0.25,
    iou: float = 0.5,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[str, dict]:
    """
    Jalankan ByteTrack tracking pada video dan render output ter-anotasi.

    Args:
        model             : YOLO model
        input_path        : path ke video input (file sementara)
        conf              : confidence threshold
        iou               : IoU threshold
        progress_callback : fungsi(float 0-1) untuk update progress bar

    Returns:
        output_path : path ke video output ter-anotasi
        stats       : dict statistik tracking
    """
    video_info = get_video_info(input_path)
    total_frames = video_info["total_frames"]
    fps = video_info["fps"]
    w, h = video_info["width"], video_info["height"]

    # Buat file output sementara
    out_fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # Jalankan tracking dengan stream=True agar bisa update progress
    tracking_results = model.track(
        source=input_path,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS],
        conf=conf,
        iou=iou,
        persist=True,   # KRITIS: jaga state tracker antar frame
        stream=True,    # Generator untuk iterasi frame-by-frame
        verbose=False,
    )

    cap = cv2.VideoCapture(input_path)
    track_history: defaultdict = defaultdict(list)
    track_frame_data = []
    frame_idx = 0

    for result in tracking_results:
        ret, frame = cap.read()
        if not ret:
            break

        boxes_arr = np.array([], dtype=int).reshape(0, 4)
        track_ids_arr = np.array([], dtype=int)
        confs_arr = np.array([], dtype=float)

        if result.boxes is not None and result.boxes.id is not None:
            boxes_arr = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids_arr = result.boxes.id.int().cpu().numpy()
            confs_arr = result.boxes.conf.cpu().numpy()

            for box, tid, conf_val in zip(boxes_arr, track_ids_arr, confs_arr):
                x1, y1, x2, y2 = box
                track_frame_data.append({
                    "frame": frame_idx,
                    "track_id": int(tid),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(conf_val),
                })

        annotated = draw_tracking_frame(
            frame, boxes_arr, track_ids_arr, confs_arr,
            track_history, frame_idx
        )
        writer.write(annotated)
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    writer.release()

    # Hitung statistik tracking
    df = pd.DataFrame(track_frame_data)
    if not df.empty:
        track_stats = df.groupby("track_id").agg(
            jumlah_frame=("frame", "count"),
            frame_awal=("frame", "min"),
            frame_akhir=("frame", "max"),
            mean_conf=("conf", "mean"),
        ).reset_index()
        track_stats["durasi"] = track_stats["frame_akhir"] - track_stats["frame_awal"] + 1
        track_stats["continuity"] = track_stats["jumlah_frame"] / track_stats["durasi"]

        stats = {
            "unique_ids": int(df["track_id"].nunique()),
            "total_frames": frame_idx,
            "total_detections": len(df),
            "avg_persons_per_frame": float(df.groupby("frame")["track_id"].nunique().mean()),
            "avg_confidence": float(df["conf"].mean()),
            "avg_continuity": float(track_stats["continuity"].mean()),
            "track_stats_df": track_stats,
            "tracking_df": df,
        }
    else:
        stats = {
            "unique_ids": 0,
            "total_frames": frame_idx,
            "total_detections": 0,
            "avg_persons_per_frame": 0.0,
            "avg_confidence": 0.0,
            "avg_continuity": 0.0,
            "track_stats_df": pd.DataFrame(),
            "tracking_df": pd.DataFrame(),
        }

    return output_path, stats
