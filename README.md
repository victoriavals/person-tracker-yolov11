# 🎯 Person Tracker YOLOv11

Sistem deteksi dan pelacakan orang (*person tracking*) berbasis deep learning menggunakan **YOLOv11** yang telah di-fine-tune secara khusus pada dataset **COCO-2017**, terintegrasi dengan algoritma *Multi-Object Tracking* **ByteTrack**, dan dilengkapi dengan antarmuka web interaktif menggunakan **Streamlit**.

---

## 🚀 Fitur Utama

1. **Deteksi Orang di Gambar**: Menggunakan model YOLOv11 yang di-fine-tune untuk mendeteksi individu secara akurat dengan bounding box dan confidence score.
2. **Video Person Tracking**: Melacak pergerakan orang dalam video dari frame ke frame menggunakan algoritma **ByteTrack**, lengkap dengan penetapan ID unik untuk tiap individu.
3. **End-to-End Data Science Pipeline**: Termasuk notebook Jupyter terpisah dari fase Akuisisi Data, Preprocessing, Training, Evaluasi, hingga Visualisasi.
4. **Berbasis Antarmuka Web (Streamlit)**: GUI interaktif yang mudah digunakan untuk mendemonstrasikan sistem tanpa perlu coding, dengan dukungan tab untuk Tracking, Detection, dan Benchmarking metrik.

---

## 🛠️ Teknologi & Environment

- **Model Utama**: YOLOv11s & YOLOv11m (`ultralytics`, `torch`)
- **Tracking Algorithm**: ByteTrack (dibantu oleh library `lap`)
- **Frontend / Aplikasi Web**: `streamlit`
- **Data & Dataset Ops**: `fiftyone` (Dataset COCO-2017), `pandas`, `numpy`
- **Image & Video Processing**: `opencv-python-headless`, `Pillow`
- **Visualisasi**: `matplotlib`, `seaborn`

---

## 📁 Struktur Direktori Proyek

Proyek ini telah direstrukturisasi agar lebih modular dan bersih, memisahkan *pipeline* riset (Notebook) dan *production* web (Streamlit).

```text
person-tracker-yolov11/
├── app.py                         # File Utama Web Streamlit
├── requirements.txt               # Daftar environment & dependency
├── .gitignore
├── utils/                         # Modul Python untuk fungsi backend (modular)
│   ├── detection.py               # Logic deteksi gambar
│   ├── tracking.py                # Logic algoritma ByteTrack untuk video
│   └── model.py                   # Load weights & model config
├── Notebooks (AI Pipeline):       # Tahapan End-to-End Data Science
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing_eda.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_tuning.ipynb
│   └── 06_visualization.ipynb
├── data/                          # Tempat output dataset COCO diunduh
├── output/                        # Hasil inferensi dan benchmarking
├── runs/                          # Metrik dan curves hasil training YOLO
└── *.pt                           # Model weights (yolo11s.pt, dll)
```

---

## 📈 Evaluasi Model

Model dalam repository ini merupakan hasil *fine-tuning* pada spesifik kelas `person` (menggunakan ~3000 gambar dari COCO-2017). Dibandingkan model pre-trained original:
- **mAP@0.5**: Meningkat **0.739** (sebelumnya 0.731)
- **mAP@0.5:0.95**: Meningkat **0.493** (sebelumnya 0.480)
- Konvergensi lebih cepat (epoch 38) dengan hyperparameter khusus (Strategy A).

---

## ⚙️ Petunjuk Instalasi & Eksekusi

### 1. Instalasi Environment
Disarankan untuk menggunakan `venv` atau `conda`. Jalankan perintah berikut untuk menginstal dependensinya:
```bash
pip install -r requirements.txt
```
> **Catatan**: Jika diperlukan paket pendukung tambahan yang tiak ada di constraints, pastikan instalasi dilakukan seperti `pip install fiftyone lap`.

### 2. Menjalankan Aplikasi Web Streamlit
Jalankan perintah ini di root direktori proyek:
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka pada browser default Anda (biasanya di alamat `http://localhost:8501`).

---

## 🧠 Menjalankan AI Pipeline (*Opsional*)

Jika Anda ingin meninjau metodologi bagaimana model ini dilatih, atau memperbarui dataset:
Anda dapat membuka file bernomor pada Jupyter Notebook (`01_data_acquisition.ipynb` sampai `06_visualization.ipynb`) secara bertahap dan menjalankan *cell* eksekusi di sana yang berisi seluruh dokumentasi kode Data Science pipeline.

---
*Proyek ini telah dioptimalkan untuk mendemonstrasikan proses AI Engineer & MLOps dasar dari pengumpulan data hingga deployment web app.*
