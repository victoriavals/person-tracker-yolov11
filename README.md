# Person Tracker — YOLOv11

A person detection and tracking system built on **YOLOv11** fine-tuned on **COCO-2017**, integrated with the **ByteTrack** multi-object tracking algorithm, and served through an interactive **Streamlit** web application.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

> **Windows note:** If you run the research notebooks, `fiftyone` requires an internal MongoDB instance. Kill any leftover `mongod.exe` process before the first run. Use `workers=0` in all `model.train()` calls to avoid CUDA subprocess spawn errors (WinError 1455).

---

## Application Features

The Streamlit app has three tabs:

| Tab | Description |
|-----|-------------|
| **Detection** | Upload an image — get bounding boxes, confidence scores, and per-detection stats |
| **Tracking** | Upload a video — ByteTrack assigns persistent IDs, draws trajectories, outputs annotated video |
| **Benchmark** | View evaluation metrics, PR curve, F1 curve, confusion matrix, and training curves for the fine-tuned model |

**Sidebar controls:** model selector, confidence threshold (default 0.25), IoU threshold (default 0.50).

Upload limit: **50 MB**. Videos over 500 frames display a processing-time warning.

---

## Models

Three models are selectable in the app:

| Option | Weights | Notes |
|--------|---------|-------|
| ✅ Custom Fine-tuned (Strategy A) | `files/runs/detect/runs/tune/strategy_A/weights/best.pt` | **Recommended** — person-specific, mAP@0.5 = 0.739 |
| yolo11n | auto-downloaded | Pretrained COCO, fastest |
| yolo11s | auto-downloaded | Pretrained COCO, balanced |

> The fine-tuned `.pt` file is not committed. Place it manually at `files/runs/detect/runs/tune/strategy_A/weights/best.pt`. Pretrained weights are downloaded automatically by Ultralytics on first use.

---

## Project Structure

```
person-tracker-yolov11/
│
├── app.py                        # Streamlit app entry point
├── requirements.txt
│
├── utils/
│   ├── model.py                  # MODEL_OPTIONS, PERSON_CLASS, load_model() with @st.cache_resource
│   ├── detection.py              # detect_image() → (annotated_bgr, stats_dict)
│   ├── tracking.py               # track_video() → ByteTrack streaming, (output_path, stats_dict)
│   └── visualization.py          # draw_detections(), draw_tracking_frame(), per-ID color palette
│
├── 01_data_acquisition.ipynb     # Download & filter COCO-2017, export to YOLO format
├── 02_preprocessing_eda.ipynb    # EDA: bbox distribution, resolution, class co-occurrence
├── 03_training.ipynb             # Fine-tune YOLOv11s — 50 epochs, 3,000 images
├── 04_evaluation.ipynb           # Test-set evaluation: mAP, PR curve, confusion matrix
├── 05_tuning.ipynb               # Hyperparameter tuning → Strategy A
├── 06_visualization.ipynb        # Hard cases, IoU distribution, trajectory visualization
│
├── output/finetune/dataset/      # YOLO-format dataset (images/ + labels/ split train/val/test)
└── files/runs/detect/runs/
    ├── train/person_yolo11s_v1/  # Initial training run (50 epochs)
    ├── eval/test_eval/           # Evaluation artifacts (PR curve, confusion matrix, hard cases)
    └── tune/strategy_A/
        └── weights/best.pt       # Best fine-tuned model — loaded by app.py
```

---

## ML Pipeline

### Dataset

- **Source:** COCO-2017 via FiftyOne Zoo, filtered to `person` class only
- **Size:** 3,000 images split 70 / 15 / 15

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 2,100 | Fine-tuning |
| Val | 450 | Training monitor |
| Test | 450 | Final evaluation (held-out) |

Labels are stored in YOLO format (`class cx cy w h`, all values 0.0–1.0). `class = 0` is always `person`.

### Training & Tuning

```
yolo11s.pt  (pretrained, 80 COCO classes)
      │
      ▼
Fine-tune — 50 epochs, imgsz=640, AdamW, freeze=10
      │
person_yolo11s_v1  (mAP@0.5: 0.731)
      │
      ▼
Hyperparameter Tuning — Strategy A
imgsz=736, aggressive augmentation, 43 epochs
      │
strategy_A/weights/best.pt  (mAP@0.5: 0.739) ✓
```

`freeze=10` freezes the first 10 backbone layers to prevent catastrophic forgetting while adapting the neck and head to person-only detection.

### Evaluation Results (Strategy A — Test Set)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.739** (+0.008 vs baseline) |
| **mAP@0.5:0.95** | **0.493** (+0.013 vs baseline) |
| Precision | 77.3% |
| Recall | 67.4% |
| Best Epoch | 38 / 43 |

---

## Tracking

ByteTrack is used for multi-object tracking:

- **No Re-ID required** — tracking is purely bounding-box-based
- **Dual-threshold** — low-confidence detections are retained to preserve IDs during occlusion
- **Kalman filter** — predicts position when a person is temporarily missed
- Enabled via `model.track(tracker="bytetrack.yaml", persist=True, stream=True)`

`persist=True` is critical — it maintains tracker state across frames.

**Per-track stats computed:**

| Stat | Description |
|------|-------------|
| `unique_ids` | Total distinct persons tracked |
| `avg_persons_per_frame` | Average crowd density |
| `continuity` | `frames_detected / active_duration` — 1.0 = perfect tracking |
| `avg_confidence` | Mean detection confidence across all tracked frames |

---

## Dependencies

```
ultralytics>=8.3.0      # YOLOv11 inference, training, ByteTrack
streamlit>=1.32.0       # Web interface
opencv-python-headless>=4.9.0
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

For the research notebooks, also install:

```bash
pip install fiftyone lap
```
