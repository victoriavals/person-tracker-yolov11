# Person Tracker YOLOv11

Sistem deteksi dan pelacakan orang (*person tracking*) berbasis deep learning menggunakan **YOLOv11** pretrained pada dataset **COCO-2017**, dengan algoritma tracking **ByteTrack**.

---

## Tujuan Proyek

Mengembangkan sistem *Person Tracking* yang mampu:
1. Mendeteksi orang dalam gambar menggunakan bounding box
2. Mengevaluasi akurasi deteksi terhadap ground truth COCO
3. Melacak identitas orang antar frame dalam video sintetis

---

## Alur Kerja

```
COCO-2017 Dataset
      ↓
Filter: Person Only
      ↓
YOLOv11 Inference (Pretrained)
      ↓
Evaluasi Deteksi (IoU, Precision, Recall, F1)
      ↓
Synthetic Video (30 frames COCO)
      ↓
ByteTrack Tracking
      ↓
Visualisasi & Output
```

---

## Hasil Evaluasi

| Metrik | Nilai |
|--------|-------|
| **Precision** | 80.4% |
| **Recall** | 73.4% |
| **F1 Score** | 0.768 |
| **Mean IoU (TP)** | 0.869 |
| **Mean Confidence (TP)** | 0.714 |
| **Mean Confidence (FP)** | 0.443 |
| **TP / FP / FN** | 611 / 149 / 221 |
| **Unique Tracking IDs** | 19 (dalam 30 frame) |

---

## Dataset

- **Sumber**: [FiftyOne Zoo — COCO-2017](https://docs.voxel51.com/dataset_zoo/datasets.html#coco-2017)
- **Split**: Validation (200 samples) + Train (300 samples)
- **Filter**: Hanya kelas `person`, semua anotasi non-person dihapus dari ground truth
- **Alasan pemilihan COCO-2017**: Versi COCO terbaru di FiftyOne Zoo, konsisten dengan distribusi training YOLOv11, memiliki 257K+ anotasi person terverifikasi

---

## Model & Tracker

| Komponen | Detail |
|----------|--------|
| **Model** | YOLOv11 Medium (`yolo11m.pt`) |
| **Weights** | Pretrained COCO (tanpa fine-tuning) |
| **Tracker** | ByteTrack (`bytetrack.yaml`) |
| **Alasan ByteTrack** | MOTA 76.1% di MOT17, tidak perlu Re-ID, dual-threshold detection, Kalman Filter untuk prediksi posisi |
| **Device** | CUDA (RTX 4060 Laptop GPU) |

---

## Struktur Project

```
person-tracker-yolov11/
├── full_executed.ipynb        # Notebook utama (sudah dieksekusi, dengan output lengkap)
├── yolo11m.pt                 # Pretrained YOLOv11 weights
├── output/
│   ├── detection/
│   │   ├── predicted_vs_groundtruth.png   # Grid 3x3: prediksi vs GT
│   │   ├── evaluation_metrics.png         # 6 subplot metrik evaluasi
│   │   └── dataset_distribution.png       # Distribusi jumlah person per gambar
│   └── tracking/
│       ├── coco_sequence.mp4              # Video sintetis dari 30 gambar COCO
│       ├── tracked_output.mp4             # Video hasil tracking dengan ID dan trajektori
│       ├── trajectory.png                 # Peta trajektori tracking
│       ├── sample_frames.png              # Contoh frame hasil tracking
│       └── tracking_statistics.png        # Statistik performa tracking
└── README.md
```

---

## Konfigurasi Utama

```python
DEVICE          = "cuda"      # GPU (RTX 4060 Laptop)
PERSON_CLASS    = 0           # Index kelas person di COCO
IOU_THRESHOLD   = 0.5         # Threshold matching prediksi ke GT
CONF_THRESHOLD  = 0.25        # Threshold confidence deteksi
VAL_SAMPLES     = 200         # Jumlah sampel validasi
TRAIN_SAMPLES   = 300         # Jumlah sampel training
SEQUENCE_LENGTH = 30          # Jumlah frame video sintetis
```

---

## Dependencies

```
ultralytics   # YOLOv11
fiftyone      # Dataset management & FiftyOne Zoo
opencv-python # Video processing
matplotlib    # Visualisasi
seaborn       # Plot styling
numpy
pandas
tqdm
lap           # ByteTrack dependency (data association)
```

Install:
```bash
pip install ultralytics fiftyone opencv-python matplotlib seaborn numpy pandas tqdm lap
```

---

## Visualisasi Output

| Warna Bounding Box | Keterangan |
|--------------------|------------|
| Hijau (solid) | Ground Truth |
| Merah (solid) | True Positive (prediksi benar) |
| Orange (solid) | False Positive (prediksi salah) |
| Biru (dashed) | False Negative (GT tidak terdeteksi) |

---

## Notebook Utama

Gunakan **`full_executed.ipynb`** — file ini berisi kode terbaru sekaligus semua hasil eksekusi (output, metrik, visualisasi).

> File ini dijalankan menggunakan `python -m nbconvert --to notebook --execute` dengan GPU CUDA.
