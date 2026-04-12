import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from PIL import Image

from utils.model import load_model, MODEL_OPTIONS
from utils.detection import detect_image, pil_to_bgr, bgr_to_rgb
from utils.tracking import track_video

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Person Tracker — YOLOv11",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    st.divider()

    # Model selector
    model_label = st.selectbox(
        "🤖 Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Pilih model. Model fine-tuned dilatih khusus pada 3.000 gambar COCO (kelas person).",
    )
    model_name = MODEL_OPTIONS[model_label]

    if model_label.startswith("✅"):
        st.success(
            "Model ini sudah **di-fine-tune** pada dataset COCO Person "
            "(3.000 gambar, 50 epoch + tuning). "
            "mAP@0.5 = **0.739**",
            icon="✅",
        )

    st.divider()

    # Threshold sliders
    conf_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Minimum confidence score untuk menampilkan deteksi.",
    )
    iou_threshold = st.slider(
        "📐 IoU Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Intersection over Union untuk Non-Maximum Suppression.",
    )

    st.divider()
    st.caption("**Person Tracker — YOLOv11s + ByteTrack**")
    st.caption("Fine-tuned pada COCO-2017 Person (3.000 gambar)")

# ─── Load Model ─────────────────────────────────────────────────────────────
model = load_model(model_name)

# ─── Header ─────────────────────────────────────────────────────────────────
st.title("🎯 Person Tracker")
st.markdown(
    "Deteksi dan tracking orang secara otomatis menggunakan **YOLOv11** + **ByteTrack**. "
    "Upload gambar atau video untuk mencoba."
)

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab_detection, tab_tracking, tab_benchmark = st.tabs(
    ["🖼️ Detection", "🎬 Tracking", "📊 Benchmark"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: DETECTION
# ════════════════════════════════════════════════════════════════════════════
with tab_detection:
    st.subheader("Deteksi Orang pada Gambar")
    st.markdown(
        "Upload gambar, model akan mendeteksi semua orang dan menampilkan "
        "bounding box dengan confidence score."
    )

    uploaded_image = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        pil_img = Image.open(uploaded_image).convert("RGB")
        img_bgr = pil_to_bgr(pil_img)

        col_orig, col_result = st.columns(2)

        with col_orig:
            st.markdown("**Gambar Original**")
            st.image(pil_img, width='stretch')

        with st.spinner("Mendeteksi orang..."):
            annotated_bgr, stats = detect_image(
                model, img_bgr, conf=conf_threshold, iou=iou_threshold
            )
            annotated_rgb = bgr_to_rgb(annotated_bgr)

        with col_result:
            st.markdown("**Hasil Deteksi**")
            st.image(annotated_rgb, width='stretch')

        st.divider()
        st.markdown("#### Statistik Deteksi")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👤 Persons Found", stats["persons_found"])
        m2.metric(
            "📈 Avg Confidence",
            f"{stats['avg_confidence']:.3f}" if stats["persons_found"] > 0 else "—",
        )
        m3.metric(
            "⬆️ Max Confidence",
            f"{stats['max_confidence']:.3f}" if stats["persons_found"] > 0 else "—",
        )
        m4.metric(
            "⬇️ Min Confidence",
            f"{stats['min_confidence']:.3f}" if stats["persons_found"] > 0 else "—",
        )

        if stats["persons_found"] == 0:
            st.info(
                "Tidak ada orang yang terdeteksi. "
                "Coba turunkan **Confidence Threshold** di sidebar.",
                icon="ℹ️",
            )

        # Download hasil
        _, buf = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            label="⬇️ Download Hasil Deteksi",
            data=buf.tobytes(),
            file_name="detection_result.jpg",
            mime="image/jpeg",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: TRACKING
# ════════════════════════════════════════════════════════════════════════════
with tab_tracking:
    st.subheader("Tracking Orang pada Video")
    st.markdown(
        "Upload video MP4, model akan melacak setiap orang dengan ID unik "
        "menggunakan algoritma **ByteTrack**."
    )

    st.info(
        "💡 **Tips:** Gunakan video pendek (< 30 detik) untuk proses lebih cepat di CPU. "
        "Batas upload: **50 MB**.",
        icon="💡",
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        # Tampilkan preview video original
        st.markdown("**Video Original**")
        st.video(uploaded_video)

        # Simpan ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name

        # Cek ukuran dan panjang video
        from utils.tracking import get_video_info
        vinfo = get_video_info(input_path)
        total_frames = vinfo["total_frames"]

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("📐 Resolusi", f"{vinfo['width']}×{vinfo['height']}")
        col_info2.metric("🎞️ Total Frame", total_frames)
        col_info3.metric("⏱️ FPS", f"{vinfo['fps']:.1f}")

        if total_frames > 500:
            st.warning(
                f"⚠️ Video ini memiliki **{total_frames} frame**. "
                "Proses mungkin membutuhkan beberapa menit di CPU.",
                icon="⚠️",
            )

        if st.button("▶️ Mulai Tracking", type="primary"):
            progress_bar = st.progress(0, text="Memproses video...")
            status_text = st.empty()

            def update_progress(val: float):
                pct = min(int(val * 100), 100)
                progress_bar.progress(pct, text=f"Memproses... {pct}%")

            with st.spinner("Menjalankan ByteTrack..."):
                output_path, stats = track_video(
                    model,
                    input_path,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    progress_callback=update_progress,
                )

            progress_bar.progress(100, text="Selesai!")
            st.success("✅ Tracking berhasil!")

            # Tampilkan video hasil
            st.markdown("**Hasil Tracking**")
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

            # Download
            st.download_button(
                label="⬇️ Download Video Tracking",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4",
            )

            st.divider()
            st.markdown("#### Statistik Tracking")

            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric("🆔 Unique IDs", stats["unique_ids"])
            ms2.metric("🎞️ Total Frame", stats["total_frames"])
            ms3.metric(
                "👥 Rata-rata Person/Frame",
                f"{stats['avg_persons_per_frame']:.1f}",
            )
            ms4.metric(
                "🎯 Avg Confidence",
                f"{stats['avg_confidence']:.3f}" if stats["unique_ids"] > 0 else "—",
            )

            if not stats["track_stats_df"].empty:
                st.markdown("#### Detail Per Track ID")
                display_df = stats["track_stats_df"][
                    ["track_id", "jumlah_frame", "frame_awal",
                     "frame_akhir", "durasi", "mean_conf", "continuity"]
                ].sort_values("durasi", ascending=False)
                display_df.columns = [
                    "Track ID", "Jumlah Frame", "Frame Awal",
                    "Frame Akhir", "Durasi", "Mean Conf", "Continuity"
                ]
                display_df["Mean Conf"] = display_df["Mean Conf"].round(3)
                display_df["Continuity"] = display_df["Continuity"].round(3)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.caption(
                    "**Continuity** mendekati 1.0 → tracking stabil. "
                    "Continuity rendah → track sering terputus (oklusi)."
                )

            # Bersihkan file sementara
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except Exception:
                pass

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.subheader("Hasil Evaluasi Model Fine-tuned")
    st.markdown(
        "Hasil evaluasi model **YOLOv11s yang sudah di-fine-tune** "
        "pada **3.000 gambar COCO-2017** (kelas `person`), "
        "dievaluasi pada **validation set** (epoch terbaik = 38). "
        "Confidence = 0.25, IoU = 0.5."
    )

    # ── Metrik Utama ──────────────────────────────────────────────────────
    st.markdown("### 🎯 Metrik Model Terbaik (Strategy A — Epoch 38)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("mAP@0.5", "0.739", help="Mean Average Precision pada IoU threshold 0.5")
    b2.metric("mAP@0.5:0.95", "0.493", help="mAP rata-rata pada berbagai IoU threshold (lebih ketat)")
    b3.metric("Precision", "77.3%", help="TP / (TP + FP) — seberapa tepat prediksi")
    b4.metric("Recall", "67.4%", help="TP / (TP + FN) — seberapa banyak objek berhasil ditemukan")

    st.divider()

    # ── Perbandingan Train vs Tune ─────────────────────────────────────────
    st.markdown("### 📈 Perbandingan: Original vs Fine-tuned (Strategy A)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Original (person_yolo11s_v1)**")
        st.markdown("""
| Metrik | Nilai |
|--------|-------|
| mAP@0.5 | 0.731 |
| mAP@0.5:0.95 | 0.477 |
| Best Epoch | 46 / 50 |
""")
    with c2:
        st.markdown("**Model Fine-tuned (Strategy A) ✅**")
        st.markdown("""
| Metrik | Nilai |
|--------|-------|
| mAP@0.5 | **0.739** (+0.008) |
| mAP@0.5:0.95 | **0.493** (+0.016) |
| Best Epoch | **38** / 43 (lebih cepat converge) |
""")

    st.divider()

    # ── Grafik Evaluasi ────────────────────────────────────────────────────
    st.markdown("### 📊 Grafik Evaluasi")

    eval_dir = Path("runs/detect/runs/eval/test_eval")
    train_dir = Path("runs/detect/runs/train/person_yolo11s_v1")
    custom_eval_dir = Path("runs/eval/test_eval")

    col1, col2 = st.columns(2)
    with col1:
        pr_img = eval_dir / "BoxPR_curve.png"
        if pr_img.exists():
            st.markdown("**Precision-Recall Curve**")
            st.image(str(pr_img), use_container_width=True)
            st.caption("Area di bawah kurva = mAP. Semakin ke kanan atas semakin baik.")

    with col2:
        f1_img = eval_dir / "BoxF1_curve.png"
        if f1_img.exists():
            st.markdown("**F1-Confidence Curve**")
            st.image(str(f1_img), use_container_width=True)
            st.caption("Titik tertinggi menunjukkan threshold confidence optimal.")

    col3, col4 = st.columns(2)
    with col3:
        cm_img = eval_dir / "confusion_matrix_normalized.png"
        if cm_img.exists():
            st.markdown("**Confusion Matrix (Test Set)**")
            st.image(str(cm_img), use_container_width=True)
            st.caption("Seberapa sering model benar/salah mendeteksi orang.")

    with col4:
        iou_img = custom_eval_dir / "iou_distribution.png"
        if iou_img.exists():
            st.markdown("**Distribusi IoU**")
            st.image(str(iou_img), use_container_width=True)
            st.caption("Seberapa akurat posisi bounding box prediksi vs ground truth.")

    # Training curves
    curves_img = train_dir / "training_curves_custom.png"
    if curves_img.exists():
        st.markdown("**Kurva Training (50 Epoch)**")
        st.image(str(curves_img), use_container_width=True)
        st.caption("Loss menurun dan mAP meningkat menandakan model berhasil belajar.")

    # Hard cases
    hard_img = custom_eval_dir / "hard_cases.png"
    if hard_img.exists():
        st.markdown("**Hard Cases — Kasus Sulit**")
        st.image(str(hard_img), use_container_width=True)
        st.caption("Contoh gambar yang sulit dideteksi: oklusi, kerumunan padat, objek kecil.")

    st.divider()

    # ── Info Training ──────────────────────────────────────────────────────
    st.markdown("### 🔧 Detail Training")
    st.markdown("""
| Properti | Detail |
|----------|--------|
| Base model | YOLOv11s (pretrained ImageNet + COCO) |
| Dataset | COCO-2017, kelas `person` saja |
| Jumlah gambar | 3.000 (Train 70% / Val 15% / Test 15%) |
| Epochs (original) | 50 epoch, imgsz=640 |
| Epochs (Strategy A) | 43 epoch (max 80 + early stop), imgsz=640, lr=0.001, patience=20 |
| Optimizer | SGD |
| Hardware | GPU (CUDA) |
| Model terbaik | `strategy_A/weights/best.pt` |
""")
