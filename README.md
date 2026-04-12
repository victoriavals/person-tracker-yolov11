# Person Tracker YOLOv11

Sistem **deteksi dan pelacakan orang** (*person tracking*) berbasis deep learning menggunakan **YOLOv11** yang di-*fine-tune* pada dataset **COCO-2017**, diintegrasikan dengan algoritma *multi-object tracking* **ByteTrack**, dan dilengkapi antarmuka web interaktif dengan **Streamlit**.

---

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Image Detection** | Deteksi orang pada gambar statis dengan bounding box dan confidence score |
| **Video Tracking** | Pelacakan orang antar-frame dengan ID unik persisten menggunakan ByteTrack |
| **Model Comparison** | Pilih antara model fine-tuned (Strategy A) atau pretrained standar |
| **Interactive Benchmark** | Visualisasi metrik evaluasi, PR curve, confusion matrix, training curves |
| **End-to-End Pipeline** | 6 notebook modular: akuisisi data → preprocessing → training → evaluasi → tuning → visualisasi |

---

## Demo Aplikasi

```bash
streamlit run app.py
```

Akses di browser: `http://localhost:8501`

**Tab yang tersedia:**
- **Detection** — upload gambar, deteksi langsung dengan statistik (jumlah orang, avg/max/min confidence)
- **Tracking** — upload video, ByteTrack dengan progress bar, tabel per-track ID, download hasil
- **Benchmark** — metrik evaluasi model fine-tuned, grafik PR curve, F1 curve, confusion matrix

---

## Instalasi

```bash
pip install -r requirements.txt
pip install fiftyone lap
```

> **Catatan Windows:** `fiftyone` memerlukan MongoDB internal — pastikan tidak ada proses `mongod.exe` yang berjalan dari sesi sebelumnya sebelum pertama kali dijalankan.

### Requirements utama

| Package | Versi | Fungsi |
|---------|-------|--------|
| `ultralytics` | ≥8.3.0 | YOLOv11 inference & training |
| `streamlit` | ≥1.32.0 | Web interface |
| `opencv-python-headless` | ≥4.9.0 | Image & video processing |
| `torch` | ≥2.2.0 | Deep learning backend |
| `fiftyone` | latest | COCO-2017 dataset management |
| `lap` | latest | ByteTrack dependency (linear assignment) |

---

## Struktur Proyek

```
person-tracker-yolov11/
│
├── app.py                          # Aplikasi web Streamlit (3 tab: Detection, Tracking, Benchmark)
├── requirements.txt
│
├── utils/                          # Backend modular
│   ├── model.py                    # load_model() dengan @st.cache_resource, MODEL_OPTIONS
│   ├── detection.py                # detect_image() → (annotated_frame, stats_dict)
│   ├── tracking.py                 # track_video() → ByteTrack streaming, (output_path, stats_dict)
│   └── visualization.py           # draw_detections(), draw_tracking_frame(), palet warna per-ID
│
├── Notebooks — Pipeline AI:
│   ├── 01_data_acquisition.ipynb  # Download & filter COCO-2017, ekspor ke format YOLO
│   ├── 02_preprocessing_eda.ipynb # EDA: distribusi person, ukuran bbox, resolusi, co-occurrence
│   ├── 03_training.ipynb          # Fine-tuning YOLOv11s (50 epoch, 3000 gambar COCO)
│   ├── 04_evaluation.ipynb        # Evaluasi pada test set: mAP, PR curve, confusion matrix
│   ├── 05_tuning.ipynb            # Hyperparameter tuning → Strategy A (mAP@0.5 0.739)
│   └── 06_visualization.ipynb     # Hard cases, IoU distribution, trajectory visualization
│
├── full.ipynb                      # Notebook lengkap (48 cell) — riset komprehensif
│
├── output/
│   ├── detection/                  # EDA plots, predicted vs GT visualization, evaluation charts
│   ├── tracking/                   # coco_sequence.mp4, tracked_output.mp4, trajectory.png
│   └── finetune/dataset/           # Dataset YOLO format: images/ + labels/ (train/val/test)
│
└── files/runs/detect/runs/
    ├── train/person_yolo11s_v1/    # Hasil training awal (50 epoch)
    ├── eval/test_eval/             # Evaluasi test set (PR curve, confusion matrix, hard cases)
    └── tune/strategy_A/
        └── weights/best.pt         # Model terbaik — digunakan oleh app.py
```

---

## Pipeline Data Science

### Dataset: COCO-2017

- **Sumber**: [FiftyOne Zoo — COCO-2017](https://docs.voxel51.com/dataset_zoo/datasets.html#coco-2017)
- **Total**: 3.000 gambar, filter kelas `person` only
- **Split 70 / 15 / 15**:

| Split | Jumlah | Fungsi |
|-------|--------|--------|
| Train | 2.100 gambar | Fine-tuning model |
| Val | 450 gambar | Monitoring saat training |
| Test | 450 gambar | Evaluasi final (tidak dilihat saat training) |

- Semua anotasi non-person dihapus dari ground truth setelah loading

### Format Label (YOLO)

```
# output/finetune/dataset/labels/train/*.txt
<class_id> <cx> <cy> <width> <height>    # nilai relatif 0.0–1.0

Contoh:
0 0.512 0.340 0.180 0.420
│  └── center x   └── lebar box
└── class 0 = person
```

---

## Model

### Varian yang Tersedia di Aplikasi

| Label | Model | Keterangan |
|-------|-------|------------|
| ✅ Custom Fine-tuned (Strategy A) | `strategy_A/weights/best.pt` | Fine-tuned khusus person — **direkomendasikan** |
| yolo11n | `yolo11n` | Pretrained COCO, 2.6M param, tercepat |
| yolo11s | `yolo11s` | Pretrained COCO, 9.4M param, seimbang |

### Proses Fine-Tuning

```
yolo11s.pt (pretrained, 80 kelas COCO)
          │
  Fine-tuning 50 epoch
  Dataset: 3.000 gambar person-only
  Optimizer: AdamW, imgsz=640
          │
  person_yolo11s_v1 (mAP@0.5: 0.731)
          │
  Hyperparameter Tuning (Strategy A)
  imgsz=736, augmentasi agresif, 43 epoch
          │
  strategy_A/weights/best.pt (mAP@0.5: 0.739) ✓
```

**Freeze strategy**: 10 layer backbone awal dibekukan (`freeze=10`) — mencegah *catastrophic forgetting* pada fitur umum, hanya mengadaptasi neck + head ke kelas person.

---

## Evaluasi Model

### Hasil Final (Strategy A — Test Set)

| Metrik | Nilai | Keterangan |
|--------|-------|------------|
| **mAP@0.5** | **0.739** | +0.008 vs baseline |
| **mAP@0.5:0.95** | **0.493** | +0.013 vs baseline |
| **Precision** | 77.3% | TP / (TP + FP) |
| **Recall** | 67.4% | TP / (TP + FN) |
| Best Epoch | 38 / 43 | Konvergensi lebih cepat |

### Perbandingan Baseline vs Fine-tuned

| | Pretrained (yolo11m) | Fine-tuned v1 | Strategy A |
|--|--|--|--|
| mAP@0.5 | 0.731 | 0.731 | **0.739** |
| mAP@0.5:0.95 | 0.480 | 0.480 | **0.493** |
| Best Epoch | — | 46 / 50 | **38 / 43** |

### Evaluasi Deteksi Pretrained (full.ipynb, 450 gambar val)

| Metrik | Nilai |
|--------|-------|
| True Positive (TP) | 611 |
| False Positive (FP) | 149 |
| False Negative (FN) | 221 |
| **Precision** | **80.4%** |
| **Recall** | **73.4%** |
| **F1-Score** | **0.768** |
| Mean IoU (TP only) | 0.869 |

---

## Tracking dengan ByteTrack

### Mengapa ByteTrack?

| Keunggulan | Detail |
|------------|--------|
| Tanpa Re-ID | Tidak butuh model pengenal tampilan visual — cukup bounding box |
| Dual-threshold | Deteksi confidence rendah tetap dipakai untuk menjaga ID saat oklusi |
| Kalman Filter | Prediksi posisi orang di frame berikutnya meski tidak terdeteksi |
| MOTA 76.1% | State-of-the-art di benchmark MOT17 tanpa Re-ID |
| Built-in | `tracker="bytetrack.yaml"` langsung di Ultralytics |

### Metrik Tracking yang Dihitung

| Metrik | Penjelasan |
|--------|------------|
| `unique_ids` | Jumlah individu berbeda yang terlacak |
| `avg_persons_per_frame` | Kepadatan rata-rata per frame |
| `continuity` | `jumlah_frame_muncul / durasi_aktif` — 1.0 = tracking sempurna |
| `avg_confidence` | Rata-rata confidence semua deteksi saat tracking |

### Visualisasi Tracking

- **Bounding box** berwarna unik per track ID (seed=42, konsisten antar sesi)
- **Label** menampilkan `ID:{tid} | {conf:.2f}`
- **Garis trajektori** di belakang setiap orang (makin baru makin tebal)
- **Overlay info** frame index + algoritma di pojok kiri atas

---

## Konfigurasi Default

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `CONF_THRESHOLD` | 0.25 | Minimum confidence untuk menampilkan deteksi |
| `IOU_THRESHOLD` | 0.50 | Threshold NMS & matching prediksi ke ground truth |
| `PERSON_CLASS` | 0 | Index kelas person di COCO (selalu 0) |
| `SEQUENCE_LENGTH` | 30 | Frame untuk synthetic tracking video di riset |
| `VAL_SAMPLES` | 450 | 15% dari 3.000 gambar (evaluasi) |
| `TRAIN_SAMPLES` | 2100 | 70% dari 3.000 gambar (EDA) |
| `TEST_SAMPLES` | 450 | 15% dari 3.000 gambar (fine-tuning test) |

---

## Legenda Warna Bounding Box (Riset)

| Warna | Status | Arti |
|-------|--------|------|
| Hijau solid | GT matched | Ground truth yang berhasil terdeteksi |
| Biru dashed | FN | Ground truth yang tidak terdeteksi |
| Merah solid | TP | Prediksi benar (IoU ≥ 0.5 dengan GT) |
| Orange solid | FP | Prediksi salah (tidak ada orang di sana) |

---

## Catatan Platform (Windows)

- **`workers=0`** wajib di semua `model.train()` — CUDA worker subprocess gagal spawn di Windows (WinError 1455)
- **FiftyOne MongoDB** — jika muncul `ServiceListenTimeout`, kill proses `mongod.exe` yang tersisa dari sesi sebelumnya
- **`dataset.yaml`** menggunakan path absolut — perlu diupdate jika repo dipindahkan ke lokasi lain
- **Model `.pt`** tidak di-commit (ada di `.gitignore`) — Ultralytics auto-download pretrained, fine-tuned perlu disediakan manual di `files/runs/detect/runs/tune/strategy_A/weights/best.pt`
