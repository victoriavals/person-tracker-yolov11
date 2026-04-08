# ===== CODE CELL [02] =====
import subprocess
import torch

print("=" * 65)
print("     GPU INFORMATION — NVIDIA System Management Interface")
print("=" * 65)
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("[WARNING] nvidia-smi tidak dapat dijalankan.")
        print(result.stderr)
except FileNotFoundError:
    print("[WARNING] nvidia-smi tidak ditemukan. Driver NVIDIA mungkin belum terinstall.")

print("=" * 65)
print("     GPU INFORMATION — PyTorch")
print("=" * 65)
print(f"  PyTorch Version      : {torch.__version__}")
print(f"  CUDA Tersedia        : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA Version         : {torch.version.cuda}")
    print(f"  Jumlah GPU           : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU [{i}] Nama         : {torch.cuda.get_device_name(i)}")
        print(f"  GPU [{i}] Total Memory : {props.total_memory / 1e9:.2f} GB")
        print(f"  GPU [{i}] Compute Cap  : {props.major}.{props.minor}")
    device = "cuda"
else:
    print("  [WARNING] GPU tidak tersedia. Menggunakan CPU.")
    device = "cpu"

print("=" * 65)
print(f"  >> Device aktif: {device.upper()}")
print("=" * 65)

# ===== CODE CELL [04] =====
# Install dependencies (jalankan sekali)
# SKIPPED (shell): !pip install ultralytics fiftyone opencv-python matplotlib seaborn numpy pandas tqdm --quiet

# ===== CODE CELL [05] =====
import os, cv2, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
from ultralytics import YOLO

import fiftyone as fo
import fiftyone.zoo as foz

# ─── Konfigurasi Global ─────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
PERSON_CLASS    = 0      # Class ID untuk 'person' di COCO (index 0)
PERSON_LABEL    = "person"  # Label string — digunakan untuk filter ketat
IOU_THRESHOLD   = 0.5    # Minimum IoU untuk dianggap True Positive
CONF_THRESHOLD  = 0.25   # Minimum confidence YOLOv11 untuk deteksi

# ─── Training / Dataset Limits ──────────────────────────────────────
# Atur di sini untuk mengontrol berapa banyak data yang dimuat.
# Dataset: COCO-2017 (versi COCO terbaru dan terlengkap di FiftyOne Zoo)
VAL_SAMPLES     = 200    # Limit gambar validation (evaluasi + visualisasi)
TRAIN_SAMPLES   = 500    # Limit gambar training (analisis distribusi data)
VIZ_SAMPLES     = 9      # Gambar yang divisualisasikan (grid 3x3)
SEQUENCE_LENGTH = 30     # Jumlah frame untuk synthetic tracking video

# Alias untuk backward compatibility
MAX_SAMPLES = VAL_SAMPLES

# ─── Direktori Output ───────────────────────────────────────────────
OUTPUT_DIR = Path("output")
(OUTPUT_DIR / "detection").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tracking").mkdir(parents=True, exist_ok=True)

import logging
logging.getLogger("fiftyone").setLevel(logging.WARNING)

print("✓ Semua library berhasil diimport")
print(f"  Device aktif      : {DEVICE}")
print(f"  Output dir        : {OUTPUT_DIR.absolute()}")
print()
print("─── Dataset Limits ─────────────────────────────────────────")
print(f"  VAL_SAMPLES       : {VAL_SAMPLES}   (gambar validation)")
print(f"  TRAIN_SAMPLES     : {TRAIN_SAMPLES} (gambar training)")
print(f"  Fokus kelas       : '{PERSON_LABEL}' ONLY")
print(f"  CONF_THRESHOLD    : {CONF_THRESHOLD}")
print(f"  IOU_THRESHOLD     : {IOU_THRESHOLD}")
print("────────────────────────────────────────────────────────────")

# ===== CODE CELL [07] =====
# ─── Load COCO-2017 Validation Split (untuk evaluasi deteksi) ────────
# Validation split digunakan karena memiliki ground truth lengkap dan terverifikasi.
# Ini adalah data yang akan dibandingkan antara prediksi vs ground truth.

print("=" * 62)
print("  LOADING DATASET: COCO-2017 (Versi Terbaru di FiftyOne Zoo)")
print("=" * 62)

# Bersihkan dataset lama jika ada
for ds_name in ["coco_val_person", "coco_train_person"]:
    if ds_name in fo.list_datasets():
        fo.delete_dataset(ds_name)

print(f"\n[1/2] Loading COCO-2017 VALIDATION split...")
print(f"      Limit    : {VAL_SAMPLES} gambar")
print(f"      Filter   : classes=['{PERSON_LABEL}'] — person ONLY")

coco_val = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],   # Hanya bounding box (bukan segmentasi/keypoints)
    classes=[PERSON_LABEL],       # Hanya muat gambar yang mengandung 'person'
    max_samples=VAL_SAMPLES,      # Limit sesuai VAL_SAMPLES
    dataset_name="coco_val_person"
)

print(f"      Loaded   : {len(coco_val)} gambar")

print(f"\n[2/2] Loading COCO-2017 TRAIN split...")
print(f"      Limit    : {TRAIN_SAMPLES} gambar")
print(f"      Filter   : classes=['{PERSON_LABEL}'] — person ONLY")

coco_train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=[PERSON_LABEL],
    max_samples=TRAIN_SAMPLES,    # Limit sesuai TRAIN_SAMPLES
    dataset_name="coco_train_person"
)

print(f"      Loaded   : {len(coco_train)} gambar")

# Alias: gunakan validation untuk evaluasi utama
coco_dataset = coco_val

print(f"\n{'=' * 62}")
print(f"  Dataset aktif untuk evaluasi : VALIDATION ({len(coco_val)} gambar)")
print(f"  Dataset untuk analisis       : TRAIN ({len(coco_train)} gambar)")
print(f"{'=' * 62}")

# ===== CODE CELL [09] =====
# ─── Filter Ketat: Hapus Semua Anotasi Non-Person ──────────────────
# FiftyOne memuat sample berdasarkan filter kelas, tapi ground truth
# satu sample bisa berisi kelas lain (co-occurring objects).
# Kita hapus SEMUA anotasi yang bukan 'person' secara eksplisit.

def filter_person_only(dataset, split_name="dataset"):
    """
    Filter ground truth: pertahankan HANYA anotasi 'person'.
    Modifikasi in-place pada dataset FiftyOne.

    Returns:
        person_counts : list jumlah person per gambar (setelah filter)
        removed_count : total anotasi non-person yang dihapus
    """
    person_counts = []
    removed_count = 0

    for sample in tqdm(dataset, desc=f"Filter person-only [{split_name}]"):
        if sample.ground_truth is None:
            person_counts.append(0)
            continue

        original_dets = sample.ground_truth.detections
        # Pertahankan HANYA bounding box dengan label 'person'
        person_dets = [d for d in original_dets if d.label == PERSON_LABEL]
        removed = len(original_dets) - len(person_dets)
        removed_count += removed

        # Update sample dengan anotasi yang sudah difilter
        sample.ground_truth.detections = person_dets
        sample.save()

        person_counts.append(len(person_dets))

    return person_counts, removed_count


print("Menerapkan filter person-only pada semua dataset...")
print()

# Filter validation split
val_counts, val_removed = filter_person_only(coco_val, "VAL")
print(f"  VAL  : {val_removed} anotasi non-person dihapus")

# Filter training split
train_counts, train_removed = filter_person_only(coco_train, "TRAIN")
print(f"  TRAIN: {train_removed} anotasi non-person dihapus")

val_counts   = np.array(val_counts)
train_counts = np.array(train_counts)

print(f"\n✓ Filter selesai! Semua ground truth kini 100% 'person' only.")
print(f"  Total anotasi non-person dihapus: {val_removed + train_removed}")

# ===== CODE CELL [10] =====
# ─── Statistik Dataset (Train + Validation) ────────────────────────

print("=" * 62)
print("  STATISTIK DATASET COCO-2017 — PERSON ONLY")
print("=" * 62)

def print_split_stats(counts, split_name, n_images):
    counts = np.array(counts)
    # Hanya hitung gambar yang punya person
    valid = counts[counts > 0]
    print(f"\n  [{split_name} Split]")
    print(f"    Gambar dimuat          : {n_images}")
    print(f"    Gambar dengan person   : {len(valid)}")
    print(f"    Total anotasi person   : {counts.sum()}")
    print(f"    Rata-rata person/gambar: {counts.mean():.2f}")
    print(f"    Min / Max per gambar   : {counts.min()} / {counts.max()}")

print_split_stats(val_counts,   "VALIDATION", len(coco_val))
print_split_stats(train_counts, "TRAIN",      len(coco_train))

print("=" * 62)

# Visualisasi distribusi kedua split secara berdampingan
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Distribusi Jumlah Person per Gambar — COCO-2017 (Person Only)",
             fontsize=13, fontweight="bold")

for ax, counts, title, color in [
    (axes[0], val_counts,   f"Validation Split (n={len(coco_val)})",   "steelblue"),
    (axes[1], train_counts, f"Train Split (n={len(coco_train)})", "mediumseagreen")
]:
    max_val = max(counts.max(), 1)
    ax.hist(counts, bins=range(0, max_val + 2),
            color=color, edgecolor="white", alpha=0.85)
    ax.axvline(counts.mean(), color="tomato", linestyle="--", linewidth=2,
               label=f"Mean = {counts.mean():.1f}")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Jumlah Person per Gambar")
    ax.set_ylabel("Jumlah Gambar")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "detection" / "dataset_distribution.png", dpi=120)
plt.show()
print("✓ Plot distribusi disimpan")

# ===== CODE CELL [12] =====
MODEL_NAME = "yolo11m.pt"  # Ganti ke yolo11n/s/l/x.pt sesuai GPU yang tersedia

print(f"Loading model: {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
model.to(DEVICE)

print("\n" + "=" * 55)
print("  INFORMASI MODEL YOLOv11")
print("=" * 55)
print(f"  Model        : {MODEL_NAME}")
print(f"  Device       : {DEVICE}")
print(f"  Total kelas  : {len(model.names)}")
print(f"  Class person : ID={PERSON_CLASS} → '{model.names[PERSON_CLASS]}'")
print("=" * 55)

assert model.names[PERSON_CLASS] == "person", "Class index salah!"
print("\n✓ Model berhasil dimuat dan siap digunakan")

# ===== CODE CELL [14] =====
def fo_bbox_to_xyxy(fo_bbox, img_w, img_h):
    """
    Konversi format FiftyOne bounding box ke format piksel absolut.

    FiftyOne: [x_top_left, y_top_left, width, height] — relatif (0.0–1.0)
    Output  : [x1, y1, x2, y2] — piksel absolut
    """
    x, y, w, h = fo_bbox
    return np.array([
        x * img_w,         # x1 (kiri)
        y * img_h,         # y1 (atas)
        (x + w) * img_w,   # x2 (kanan)
        (y + h) * img_h    # y2 (bawah)
    ])


def compute_iou(box1, box2):
    """
    Hitung IoU antara dua bounding box.

    Args:
        box1, box2 : array [x1, y1, x2, y2] dalam piksel

    Returns:
        float : IoU dalam range [0.0, 1.0]

    Cara kerja:
        1. Cari area intersection (tumpang tindih)
        2. Cari area union (gabungan)
        3. IoU = intersection / union
    """
    # Intersection: koordinat tumpang tindih
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    # Area masing-masing box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union = area1 + area2 - intersection (hindari double-count)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_preds_to_gt(pred_boxes, pred_confs, gt_boxes, iou_thresh=0.5):
    """
    Greedy matching: cocokkan setiap prediksi ke ground truth terbaik.

    Algoritma:
    1. Urutkan prediksi dari confidence tertinggi ke terendah
    2. Untuk setiap prediksi, cari GT dengan IoU tertinggi
    3. Jika IoU >= threshold: TP (tandai GT sebagai sudah dipakai)
    4. Jika IoU < threshold: FP

    Args:
        pred_boxes  : list of [x1,y1,x2,y2] — bounding box prediksi
        pred_confs  : list of float — confidence score prediksi
        gt_boxes    : list of [x1,y1,x2,y2] — ground truth boxes
        iou_thresh  : float — threshold IoU untuk TP

    Returns:
        matches        : list of (pred_idx, gt_idx, iou_value)
        unmatched_pred : list pred_idx yang FP
        unmatched_gt   : list gt_idx yang FN
    """
    if not pred_boxes or not gt_boxes:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    # Urutkan prediksi: confidence tertinggi diprioritaskan
    order = np.argsort(pred_confs)[::-1]
    matched_gt, matches, unmatched_pred = set(), [], []

    for pi in order:
        best_iou, best_gi = 0.0, -1

        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue  # GT sudah dipakai prediksi lain
            iou = compute_iou(pred_boxes[pi], gb)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_iou >= iou_thresh:
            matches.append((pi, best_gi, best_iou))
            matched_gt.add(best_gi)
        else:
            unmatched_pred.append(pi)

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


print("✓ Helper functions siap:")
print("  fo_bbox_to_xyxy()   — konversi koordinat FiftyOne → piksel")
print("  compute_iou()       — kalkulasi Intersection over Union")
print("  match_preds_to_gt() — greedy matching prediksi ↔ ground truth")

# ===== CODE CELL [16] =====
eval_results = []  # Untuk kalkulasi metrik
viz_data     = []  # Untuk visualisasi (VIZ_SAMPLES gambar pertama)

print(f"Menjalankan inferensi pada {len(coco_dataset)} gambar COCO...")
print(f"  Confidence threshold : {CONF_THRESHOLD}")
print(f"  IoU threshold        : {IOU_THRESHOLD}")
print()

for idx, sample in enumerate(tqdm(coco_dataset, desc="Inference")):
    img_path = sample.filepath
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Ground Truth dari COCO anotasi ──────────────────────────
    gt_boxes = []
    if sample.ground_truth is not None:
        for det in sample.ground_truth.detections:
            if det.label == "person":
                gt_boxes.append(fo_bbox_to_xyxy(det.bounding_box, img_w, img_h))

    # ── Prediksi YOLOv11 ────────────────────────────────────────
    results = model.predict(
        source=img_path,
        classes=[PERSON_CLASS],   # Hanya deteksi person
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False             # Matikan output verbose per gambar
    )

    pred_boxes, pred_confs = [], []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls.item()) == PERSON_CLASS:
                pred_boxes.append(box.xyxy[0].cpu().numpy())
                pred_confs.append(float(box.conf.item()))

    # ── Matching prediksi ↔ ground truth ────────────────────────
    matches, fp_list, fn_list = match_preds_to_gt(
        pred_boxes, pred_confs, gt_boxes, IOU_THRESHOLD
    )

    # Rekam hasil
    for pi, gi, iou_val in matches:
        eval_results.append({"status": "TP", "iou": iou_val,
                             "confidence": pred_confs[pi]})
    for pi in fp_list:
        eval_results.append({"status": "FP", "iou": 0.0,
                             "confidence": pred_confs[pi]})
    for _ in fn_list:
        eval_results.append({"status": "FN", "iou": 0.0, "confidence": 0.0})

    # Simpan data visualisasi
    if len(viz_data) < VIZ_SAMPLES:
        viz_data.append({
            "img_rgb":    img_rgb,
            "gt_boxes":   gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_confs": pred_confs,
            "matches":    matches,
            "fp_list":    fp_list,
            "fn_list":    fn_list
        })

df_eval = pd.DataFrame(eval_results)
print(f"\n✓ Inferensi selesai!")
print(f"  Total deteksi: {len(df_eval)}")
print(df_eval["status"].value_counts().to_string())

# ===== CODE CELL [18] =====
def draw_comparison(ax, entry):
    """
    Gambar satu gambar dengan overlay bounding box GT dan prediksi.
    GT ditampilkan berbeda untuk matched (hijau) dan missed (biru dash).
    Prediksi ditampilkan berbeda untuk TP (merah) dan FP (oranye).
    """
    img_rgb    = entry["img_rgb"]
    gt_boxes   = entry["gt_boxes"]
    pred_boxes = entry["pred_boxes"]
    pred_confs = entry["pred_confs"]
    matches    = entry["matches"]
    fp_list    = entry["fp_list"]

    ax.imshow(img_rgb)
    ax.axis("off")

    matched_pred_idx = {m[0] for m in matches}
    matched_gt_idx   = {m[1] for m in matches}

    # ── Ground Truth boxes ──────────────────────────────────────
    for gi, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = gt
        if gi in matched_gt_idx:
            color, ls, label = "lime", "-", "GT"
        else:
            color, ls, label = "dodgerblue", "--", "FN"  # Tidak terdeteksi

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none", linestyle=ls
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, label, color=color, fontsize=5.5, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.1"))

    # ── Prediksi boxes ──────────────────────────────────────────
    for pi, pb in enumerate(pred_boxes):
        x1, y1, x2, y2 = pb
        conf = pred_confs[pi]
        if pi in matched_pred_idx:
            color, label = "red", f"TP {conf:.2f}"
        else:
            color, label = "orange", f"FP {conf:.2f}"

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y2 + 8, label, color=color, fontsize=5.5, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.1"))


# Grid 3x3
n_cols = 3
n_rows = (VIZ_SAMPLES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
axes = axes.flatten()

for i, entry in enumerate(viz_data):
    draw_comparison(axes[i], entry)
    ious    = [m[2] for m in entry["matches"]]
    avg_iou = np.mean(ious) if ious else 0.0
    tp_n = len(entry["matches"])
    fp_n = len(entry["fp_list"])
    fn_n = len(entry["fn_list"])
    axes[i].set_title(
        f"TP={tp_n}  FP={fp_n}  FN={fn_n}  mIoU={avg_iou:.2f}",
        fontsize=9
    )

for j in range(len(viz_data), len(axes)):
    axes[j].set_visible(False)

# Legend
legend_elements = [
    patches.Patch(facecolor="none", edgecolor="lime",
                  linewidth=2, label="Ground Truth (Matched)"),
    patches.Patch(facecolor="none", edgecolor="dodgerblue",
                  linewidth=2, linestyle="--", label="Ground Truth (Missed / FN)"),
    patches.Patch(facecolor="none", edgecolor="red",
                  linewidth=2, label="Prediksi (True Positive)"),
    patches.Patch(facecolor="none", edgecolor="orange",
                  linewidth=2, label="Prediksi (False Positive)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("YOLOv11 Person Detection — Predicted vs Ground Truth",
             fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "detection" / "predicted_vs_groundtruth.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("✓ Disimpan: output/detection/predicted_vs_groundtruth.png")

# ===== CODE CELL [20] =====
tp_df = df_eval[df_eval["status"] == "TP"]
fp_df = df_eval[df_eval["status"] == "FP"]
fn_df = df_eval[df_eval["status"] == "FN"]

TP = len(tp_df)
FP = len(fp_df)
FN = len(fn_df)

precision    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall       = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1           = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
mean_iou     = tp_df["iou"].mean()        if TP > 0 else 0.0
mean_conf_tp = tp_df["confidence"].mean() if TP > 0 else 0.0
mean_conf_fp = fp_df["confidence"].mean() if FP > 0 else 0.0

print("=" * 58)
print("  HASIL EVALUASI DETEKSI — YOLOv11 Person Detection")
print("=" * 58)
print(f"  True Positive  (TP) : {TP:>5} — prediksi benar")
print(f"  False Positive (FP) : {FP:>5} — prediksi salah / ghost")
print(f"  False Negative (FN) : {FN:>5} — orang tidak terdeteksi")
print("-" * 58)
print(f"  Precision           : {precision:.4f}  ({precision*100:.1f}%)")
print(f"  Recall              : {recall:.4f}  ({recall*100:.1f}%)")
print(f"  F1-Score            : {f1:.4f}")
print("-" * 58)
print(f"  Mean IoU  (TP only) : {mean_iou:.4f}")
print(f"  Mean Conf (TP)      : {mean_conf_tp:.4f}")
print(f"  Mean Conf (FP)      : {mean_conf_fp:.4f}")
print("=" * 58)

# ===== CODE CELL [21] =====
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Evaluasi Metrik — YOLOv11 Person Detection",
             fontsize=14, fontweight="bold")

# ── Plot 1: Distribusi IoU (TP saja) ────────────────────────────
ax = axes[0, 0]
if TP > 0:
    ax.hist(tp_df["iou"], bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_iou, color="tomato", linestyle="--", linewidth=2,
               label=f"Mean = {mean_iou:.3f}")
    ax.axvline(IOU_THRESHOLD, color="green", linestyle=":", linewidth=2,
               label=f"Threshold = {IOU_THRESHOLD}")
ax.set_title("Distribusi IoU (True Positives)", fontweight="bold")
ax.set_xlabel("IoU Score"); ax.set_ylabel("Jumlah")
ax.set_xlim(0, 1); ax.legend(fontsize=9)

# ── Plot 2: Distribusi Confidence (TP vs FP) ────────────────────
ax = axes[0, 1]
if TP > 0:
    ax.hist(tp_df["confidence"], bins=20, color="steelblue",
            edgecolor="white", alpha=0.75, label="TP")
if FP > 0:
    ax.hist(fp_df["confidence"], bins=20, color="tomato",
            edgecolor="white", alpha=0.75, label="FP")
ax.set_title("Confidence Score: TP vs FP", fontweight="bold")
ax.set_xlabel("Confidence"); ax.set_ylabel("Jumlah")
ax.set_xlim(0, 1); ax.legend(fontsize=9)

# ── Plot 3: Scatter Confidence vs IoU ───────────────────────────
ax = axes[0, 2]
if TP > 0:
    sc = ax.scatter(tp_df["confidence"], tp_df["iou"],
                    c=tp_df["iou"], cmap="RdYlGn", alpha=0.5, s=15, vmin=0.5, vmax=1.0)
    plt.colorbar(sc, ax=ax, label="IoU")
ax.axhline(IOU_THRESHOLD, color="tomato", linestyle="--", linewidth=1.5,
           label=f"IoU={IOU_THRESHOLD}")
ax.axvline(CONF_THRESHOLD, color="dodgerblue", linestyle=":", linewidth=1.5,
           label=f"Conf={CONF_THRESHOLD}")
ax.set_title("Confidence vs IoU (True Positives)", fontweight="bold")
ax.set_xlabel("Confidence"); ax.set_ylabel("IoU")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9)

# ── Plot 4: Bar TP / FP / FN ────────────────────────────────────
ax = axes[1, 0]
bars = ax.bar(["True Positive", "False Positive", "False Negative"],
              [TP, FP, FN],
              color=["steelblue", "tomato", "orange"],
              edgecolor="white", alpha=0.85)
for b, v in zip(bars, [TP, FP, FN]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            str(v), ha="center", fontweight="bold", fontsize=12)
ax.set_title("Jumlah TP / FP / FN", fontweight="bold")
ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=10)

# ── Plot 5: Precision / Recall / F1 ─────────────────────────────
ax = axes[1, 1]
metrics   = ["Precision", "Recall", "F1-Score"]
vals      = [precision, recall, f1]
bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]
bars = ax.bar(metrics, vals, color=bar_colors, edgecolor="white", alpha=0.9)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{v:.3f}", ha="center", fontweight="bold", fontsize=13)
ax.set_ylim(0, 1.15)
ax.set_title("Precision / Recall / F1-Score", fontweight="bold")
ax.set_ylabel("Score")

# ── Plot 6: IoU per rentang ──────────────────────────────────────
ax = axes[1, 2]
iou_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
if TP > 0:
    counts, _ = np.histogram(tp_df["iou"], bins=iou_bins)
    labels = [f"{iou_bins[i]:.1f}–{iou_bins[i+1]:.1f}" for i in range(len(counts))]
    ax.bar(labels, counts, color="mediumseagreen", edgecolor="white", alpha=0.85)
ax.set_title("Distribusi IoU per Rentang (TP Only)", fontweight="bold")
ax.set_xlabel("IoU Range"); ax.set_ylabel("Jumlah TP")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "detection" / "evaluation_metrics.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("✓ Plot metrik disimpan: output/detection/evaluation_metrics.png")

# ===== CODE CELL [22] =====
# Analisis pengaruh confidence threshold terhadap metrik
total_gt  = TP + FN  # Total ground truth = TP + FN
all_preds = df_eval[df_eval["status"].isin(["TP", "FP"])].copy()

rows = []
for thresh in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    f = all_preds[all_preds["confidence"] >= thresh]
    tp_t = len(f[f["status"] == "TP"])
    fp_t = len(f[f["status"] == "FP"])
    fn_t = total_gt - tp_t
    p_t  = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    r_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) > 0 else 0
    iou_t = f[f["status"] == "TP"]["iou"].mean()
    rows.append({
        "Conf Threshold": f"{thresh:.2f}",
        "TP": tp_t, "FP": fp_t, "FN": fn_t,
        "Precision": f"{p_t:.3f}", "Recall": f"{r_t:.3f}",
        "F1-Score":  f"{f1_t:.3f}",
        "Mean IoU":  f"{iou_t:.3f}" if not np.isnan(iou_t) else "N/A"
    })

print("=" * 78)
print("  PENGARUH CONFIDENCE THRESHOLD TERHADAP METRIK")
print("=" * 78)
print(pd.DataFrame(rows).to_string(index=False))
print()
print("  Insight: Threshold ↑ → Precision ↑ (lebih sedikit FP)")
print("           Threshold ↑ → Recall ↓ (lebih banyak FN)")
print("           Pilih threshold yang seimbangkan kebutuhan aplikasi")

# ===== CODE CELL [24] =====
# ─── Step 1: Siapkan Synthetic Video dari COCO Images ───────────────

print("Menyiapkan image sequence dari COCO...")

# Kumpulkan semua gambar yang mengandung orang
image_infos = []
for sample in coco_dataset:
    img_path = sample.filepath
    if not os.path.exists(img_path) or sample.ground_truth is None:
        continue
    n_persons = len([d for d in sample.ground_truth.detections
                     if d.label == "person"])
    if n_persons > 0:
        image_infos.append({"path": img_path, "n_persons": n_persons})

# Urutkan berdasarkan jumlah person (scene paling ramai dulu)
# Ini membuat sequence lebih menarik untuk tracking demo
image_infos = sorted(image_infos, key=lambda x: -x["n_persons"])

seq_len = min(SEQUENCE_LENGTH, len(image_infos))
sequence_images = image_infos[:seq_len]

print(f"  Gambar tersedia       : {len(image_infos)}")
print(f"  Gambar untuk sequence : {seq_len}")
print(f"  Rata-rata person/frame: {np.mean([x['n_persons'] for x in sequence_images]):.1f}")

# Resize semua gambar ke ukuran seragam
TARGET_W, TARGET_H = 1280, 720
VIDEO_FPS = 3  # 3 FPS — cukup lambat agar ByteTrack bisa beradaptasi antar gambar

frames = []
for info in tqdm(sequence_images, desc="Menyiapkan frames"):
    img = cv2.imread(info["path"])
    if img is None:
        continue
    frames.append(cv2.resize(img, (TARGET_W, TARGET_H)))

# Tulis sebagai video MP4
VIDEO_PATH = str(OUTPUT_DIR / "tracking" / "coco_sequence.mp4")
writer = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    VIDEO_FPS,
    (TARGET_W, TARGET_H)
)
for f in frames:
    writer.write(f)
writer.release()

print(f"\n✓ Synthetic video dibuat!")
print(f"  Path      : {VIDEO_PATH}")
print(f"  Resolusi  : {TARGET_W} x {TARGET_H}")
print(f"  FPS       : {VIDEO_FPS}")
print(f"  Jumlah frame: {len(frames)}")
print(f"  Durasi    : {len(frames)/VIDEO_FPS:.1f} detik")

# ===== CODE CELL [25] =====
# ─── Step 2: Jalankan ByteTrack via YOLOv11 ─────────────────────────

print("Menjalankan Person Tracking: YOLOv11 + ByteTrack...")
print(f"  Tracker    : ByteTrack")
print(f"  Source     : {VIDEO_PATH}")
print(f"  Kelas      : person (class_id=0)")
print(f"  Confidence : {CONF_THRESHOLD}")
print(f"  Device     : {DEVICE}")
print()

# model.track() memproses video frame per frame
# - tracker="bytetrack.yaml" : gunakan ByteTrack
# - persist=True : PENTING — pertahankan state tracker antar frame
#   Tanpa ini, ID akan reset di setiap frame dan tracking tidak berfungsi
tracking_results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    classes=[PERSON_CLASS],
    conf=CONF_THRESHOLD,
    device=DEVICE,
    persist=True,
    save=False,
    verbose=False
)

print(f"✓ Tracking selesai! Frame diproses: {len(tracking_results)}")

# ===== CODE CELL [26] =====
# ─── Step 3: Render Video Output dengan Visualisasi Tracking ─────────

# Palet warna konsisten per track_id (random seed tetap)
rng_color = np.random.default_rng(42)
id_colors = {}

def get_color(tid):
    """Warna RGB konsisten per track ID"""
    if tid not in id_colors:
        id_colors[tid] = tuple(rng_color.integers(80, 220, size=3).tolist())
    return id_colors[tid]

track_history    = defaultdict(list)  # {track_id: [(cx,cy), ...]} untuk trajectory
track_frame_data = []                  # Rekam data per deteksi untuk analisis

cap = cv2.VideoCapture(VIDEO_PATH)
OUTPUT_VIDEO = str(OUTPUT_DIR / "tracking" / "tracked_output.mp4")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    VIDEO_FPS,
    (TARGET_W, TARGET_H)
)

print("Rendering video output tracking...")
for frame_idx, result in enumerate(tqdm(tracking_results, desc="Render")):
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    if result.boxes is not None and result.boxes.id is not None:
        boxes     = result.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = result.boxes.id.int().cpu().numpy()
        confs     = result.boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_history[int(tid)].append((cx, cy))
            track_frame_data.append({
                "frame": frame_idx, "track_id": int(tid),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(conf)
            })

            color = get_color(int(tid))

            # Bounding box dengan warna unik per ID
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label: Track ID + Confidence score
            label = f"ID:{tid} | {conf:.2f}"
            lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lsz[1] - 6),
                         (x1 + lsz[0] + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Trajectory: jalur pergerakan center point
            hist = track_history[int(tid)]
            for i in range(1, len(hist)):
                alpha_val = i / len(hist)  # Makin terang = makin baru
                cv2.line(annotated, hist[i-1], hist[i],
                         color, max(1, int(alpha_val * 3)))

    # Info overlay: nomor frame + algoritma
    cv2.putText(annotated, f"Frame {frame_idx} | ByteTrack | YOLOv11",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    writer.write(annotated)

cap.release()
writer.release()

df_tracking = pd.DataFrame(track_frame_data)
print(f"\n✓ Video output disimpan: {OUTPUT_VIDEO}")
print(f"  Total deteksi tracked: {len(df_tracking)}")
if len(df_tracking) > 0:
    print(f"  Unique Person IDs   : {df_tracking['track_id'].nunique()}")

# ===== CODE CELL [28] =====
# Sample frames dari video output tracking
print("Mengambil sample frames dari video tracking...")
cap = cv2.VideoCapture(OUTPUT_VIDEO)
total_out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_indices = np.linspace(0, total_out_frames - 1, min(6, total_out_frames), dtype=int)
sample_frames_rgb = []

for fi in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ret, frm = cap.read()
    if ret:
        sample_frames_rgb.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
cap.release()

# Grid visualisasi 2x3
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle("Sample Frame — Tracking Output (YOLOv11 + ByteTrack)",
             fontsize=14, fontweight="bold")

for i, (frm, fi) in enumerate(zip(sample_frames_rgb, sample_indices)):
    axes[i].imshow(frm)
    axes[i].set_title(f"Frame {fi}", fontsize=10)
    axes[i].axis("off")

for j in range(len(sample_frames_rgb), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tracking" / "sample_frames.png",
            dpi=100, bbox_inches="tight")
plt.show()
print("✓ Sample frames tracking disimpan")

# ===== CODE CELL [29] =====
if len(df_tracking) > 0:
    # Hitung statistik per track ID
    track_stats = df_tracking.groupby("track_id").agg(
        jumlah_frame=("frame", "count"),
        frame_awal=("frame", "min"),
        frame_akhir=("frame", "max"),
        mean_conf=("conf", "mean"),
        min_conf=("conf", "min"),
        max_conf=("conf", "max")
    ).reset_index()

    track_stats["durasi"] = (track_stats["frame_akhir"]
                              - track_stats["frame_awal"] + 1)
    # Continuity: berapa % frame dalam durasi track yang benar-benar terdeteksi
    # 1.0 = sempurna (tidak ada frame yang hilang)
    track_stats["continuity"] = (track_stats["jumlah_frame"]
                                  / track_stats["durasi"])

    print("=" * 72)
    print("  STATISTIK TRACKING PER PERSON ID")
    print("=" * 72)
    print(track_stats[["track_id", "jumlah_frame", "frame_awal",
                        "frame_akhir", "durasi", "mean_conf", "continuity"]]
          .sort_values("jumlah_frame", ascending=False).to_string(index=False))

    best_id = track_stats.loc[track_stats["durasi"].idxmax(), "track_id"]
    print("\n" + "=" * 72)
    print("  RINGKASAN GLOBAL")
    print("=" * 72)
    print(f"  Unique Person IDs        : {track_stats['track_id'].nunique()}")
    print(f"  Rata-rata durasi track   : {track_stats['durasi'].mean():.1f} frame")
    print(f"  Track terpanjang         : {track_stats['durasi'].max()} frame (ID={best_id})")
    print(f"  Rata-rata confidence     : {df_tracking['conf'].mean():.4f}")
    print(f"  Rata-rata continuity     : {track_stats['continuity'].mean():.4f}")
    print("=" * 72)
    print()
    print("  Continuity mendekati 1.0 → tracking stabil, ID tidak sering hilang")
    print("  Continuity rendah → track sering interrupted (banyak oklusi/FN)")
else:
    print("[INFO] Tidak ada person yang terdeteksi pada video sequence ini.")

# ===== CODE CELL [30] =====
if len(df_tracking) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Analisis Tracking — ByteTrack + YOLOv11",
                 fontsize=13, fontweight="bold")

    # Plot 1: Jumlah person aktif per frame
    ax = axes[0]
    ppf = df_tracking.groupby("frame")["track_id"].nunique()
    ax.plot(ppf.index, ppf.values, color="steelblue", linewidth=1.5, alpha=0.85)
    ax.fill_between(ppf.index, ppf.values, alpha=0.2, color="steelblue")
    ax.axhline(ppf.mean(), color="tomato", linestyle="--", linewidth=1.5,
               label=f"Mean = {ppf.mean():.1f}")
    ax.set_title("Jumlah Person Aktif per Frame", fontweight="bold")
    ax.set_xlabel("Frame Index"); ax.set_ylabel("Jumlah Person")
    ax.legend(fontsize=9)

    # Plot 2: Distribusi confidence (tracking)
    ax = axes[1]
    ax.hist(df_tracking["conf"], bins=30, color="mediumseagreen",
            edgecolor="white", alpha=0.85)
    ax.axvline(df_tracking["conf"].mean(), color="tomato", linestyle="--",
               linewidth=2, label=f"Mean = {df_tracking['conf'].mean():.3f}")
    ax.set_title("Distribusi Confidence Score (Tracking)", fontweight="bold")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Frekuensi")
    ax.legend(fontsize=9); ax.set_xlim(0, 1)

    # Plot 3: Gantt-style timeline per track ID
    ax = axes[2]
    top_tracks = track_stats.nlargest(min(12, len(track_stats)), "durasi")
    for _, row in top_tracks.iterrows():
        c = [x/255 for x in get_color(int(row["track_id"]))]
        ax.barh(f"ID {int(row['track_id'])}", row["durasi"],
                left=row["frame_awal"], color=c, alpha=0.8, height=0.6)
    ax.set_title("Timeline Track (Gantt-style)", fontweight="bold")
    ax.set_xlabel("Frame Index"); ax.set_ylabel("Person ID")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tracking" / "tracking_statistics.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("✓ Plot statistik tracking disimpan")

# ===== CODE CELL [31] =====
# Visualisasi Trajectory pada background frame
if track_history:
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, len(frames) // 2)
    ret, bg = cap.read()
    cap.release()

    if ret:
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(bg_rgb, alpha=0.5)  # Background semi-transparan

        for tid, positions in track_history.items():
            if len(positions) < 2:
                continue
            color = [c/255 for c in get_color(int(tid))]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            # Gambar jalur dengan gradasi (makin terang = makin baru)
            for i in range(1, len(xs)):
                alpha_val = 0.3 + 0.7 * (i / len(xs))
                ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                        color=color, alpha=alpha_val, linewidth=2.5)

            # Titik posisi terakhir (center terkini)
            ax.scatter(xs[-1], ys[-1], c=[color], s=80, zorder=5,
                       edgecolors="white", linewidths=1.5)
            ax.annotate(f"ID:{tid}", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, -5),
                        fontsize=8.5, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=color, alpha=0.9))

        ax.set_title("Trajectory Pergerakan Person — COCO Image Sequence",
                     fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tracking" / "trajectory.png",
                    dpi=120, bbox_inches="tight")
        plt.show()
        print("✓ Trajectory disimpan: output/tracking/trajectory.png")

# ===== CODE CELL [33] =====
print("=" * 65)
print("      RINGKASAN AKHIR — Person Tracker YOLOv11 + ByteTrack")
print("=" * 65)

print("\n[KONFIGURASI]")
print(f"  Model                  : {MODEL_NAME}")
print(f"  Tracker                : ByteTrack")
print(f"  Device                 : {DEVICE}")
print(f"  Confidence Threshold   : {CONF_THRESHOLD}")
print(f"  IoU Threshold          : {IOU_THRESHOLD}")
print(f"  Person Class ID        : {PERSON_CLASS} ('{PERSON_LABEL}')")
print(f"  Filter kelas           : {PERSON_LABEL} ONLY (non-person dihapus)")

print("\n[DATASET — COCO-2017 (Versi Terbaru, FiftyOne Zoo)]")
print(f"  Dataset                : COCO-2017 (train + validation)")
print(f"  VAL_SAMPLES (limit)    : {VAL_SAMPLES}  gambar → evaluasi deteksi")
print(f"  TRAIN_SAMPLES (limit)  : {TRAIN_SAMPLES} gambar → analisis distribusi")
print(f"  Filter                 : person-only (anotasi non-person dihapus)")
print(f"  Val loaded             : {len(coco_val)} gambar")
print(f"  Train loaded           : {len(coco_train)} gambar")

print("\n[EVALUASI DETEKSI — COCO-2017 Validation]")
print(f"  True Positive  (TP)    : {TP}")
print(f"  False Positive (FP)    : {FP}")
print(f"  False Negative (FN)    : {FN}")
print(f"  Precision              : {precision:.4f}  ({precision*100:.1f}%)")
print(f"  Recall                 : {recall:.4f}  ({recall*100:.1f}%)")
print(f"  F1-Score               : {f1:.4f}")
print(f"  Mean IoU (TP only)     : {mean_iou:.4f}")
print(f"  Mean Conf (TP)         : {mean_conf_tp:.4f}")
print(f"  Mean Conf (FP)         : {mean_conf_fp:.4f}")

if len(df_tracking) > 0:
    print("\n[EVALUASI TRACKING — COCO Image Sequence]")
    print(f"  Frames diproses        : {len(tracking_results)}")
    print(f"  Unique Person IDs      : {df_tracking['track_id'].nunique()}")
    print(f"  Rata-rata Confidence   : {df_tracking['conf'].mean():.4f}")
    if len(track_stats) > 0:
        print(f"  Rata-rata Track Durasi : {track_stats['durasi'].mean():.1f} frame")
        print(f"  Rata-rata Continuity   : {track_stats['continuity'].mean():.4f}")

print("\n[OUTPUT FILES]")
output_dir = Path("output")
for fp_out in sorted(output_dir.rglob("*")):
    if fp_out.is_file():
        sz = fp_out.stat().st_size / 1024
        print(f"  {str(fp_out.relative_to(output_dir)):55s} [{sz:.1f} KB]")

print("\n" + "=" * 65)
print("  Semua proses selesai!")
print("=" * 65)

