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
        help="yolo11n paling cepat dan aman untuk Streamlit Cloud.",
    )
    model_name = MODEL_OPTIONS[model_label]

    # Warning untuk model besar di Streamlit Cloud
    if model_name in ("yolo11m",):
        st.warning(
            "⚠️ **yolo11m** membutuhkan RAM lebih besar (~600MB+). "
            "Mungkin crash di Streamlit Cloud (limit 1GB).",
            icon="⚠️",
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
    st.caption("**Person Tracker — YOLOv11 + ByteTrack**")
    st.caption("Model pretrained COCO-2017 (80 classes)")

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
    st.subheader("Hasil Evaluasi Model (Pre-computed)")
    st.markdown(
        "Hasil evaluasi dihitung pada **200 gambar COCO-2017 validation set** "
        "menggunakan model **yolo11m** dengan threshold confidence=0.25, IoU=0.5."
    )

    # ── Metrik Deteksi ────────────────────────────────────────────────────
    st.markdown("### 🎯 Metrik Deteksi")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Precision", "80.4%", help="TP / (TP + FP)")
    b2.metric("Recall", "73.4%", help="TP / (TP + FN)")
    b3.metric("F1 Score", "0.768", help="2 × Precision × Recall / (Precision + Recall)")
    b4.metric("Mean IoU (TP)", "0.869", help="Rata-rata IoU untuk True Positive")

    st.divider()

    # ── Metrik Tracking ───────────────────────────────────────────────────
    st.markdown("### 🎬 Metrik Tracking (ByteTrack)")
    t1, t2, t3 = st.columns(3)
    t1.metric("Unique Person IDs", "19", help="Dalam 30 frame video sintetis")
    t2.metric("Total Frame", "30", help="Video sintetis dari COCO images")
    t3.metric("Avg Confidence (Tracking)", "0.776")

    st.divider()

    # ── Grafik dari output/ ────────────────────────────────────────────────
    st.markdown("### 📊 Visualisasi Hasil")

    output_dir = Path("output")

    col_det, col_track = st.columns(2)

    eval_img = output_dir / "detection" / "evaluation_metrics.png"
    tracking_img = output_dir / "tracking" / "tracking_statistics.png"
    sample_frames = output_dir / "tracking" / "sample_frames.png"
    trajectory_img = output_dir / "tracking" / "trajectory.png"

    with col_det:
        if eval_img.exists():
            st.markdown("**Evaluasi Deteksi (200 gambar COCO)**")
            st.image(str(eval_img), width='stretch')
        else:
            st.info("Jalankan notebook `full.ipynb` untuk menghasilkan grafik evaluasi.")

    with col_track:
        if tracking_img.exists():
            st.markdown("**Analisis Tracking (ByteTrack)**")
            st.image(str(tracking_img), width='stretch')
        else:
            st.info("Jalankan notebook `full.ipynb` untuk menghasilkan grafik tracking.")

    if sample_frames.exists():
        st.markdown("**Sample Frames Tracking**")
        st.image(str(sample_frames), width='stretch')

    if trajectory_img.exists():
        st.markdown("**Trajectory Pergerakan Person**")
        st.image(str(trajectory_img), width='stretch')

    st.divider()

    # ── Info Model ─────────────────────────────────────────────────────────
    st.markdown("### 🤖 Perbandingan Model YOLOv11")
    import pandas as pd
    df_models = pd.DataFrame({
        "Model": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        "Params": ["2.6M", "9.4M", "20.1M", "25.3M", "56.9M"],
        "mAP50-95": ["39.5%", "47.0%", "51.5%", "53.4%", "54.7%"],
        "Speed CPU (ms/img)": ["~56", "~90", "~183", "~238", "~462"],
        "Digunakan": ["", "", "✅ (Benchmark)", "", ""],
    })
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    st.markdown("### 📚 Dataset")
    st.markdown(
        """
| Properti | Detail |
|----------|--------|
| Dataset | COCO-2017 (Microsoft) |
| Split digunakan | Validation set |
| Jumlah gambar evaluasi | 200 gambar |
| Kelas target | Person (class ID: 0) |
| Total anotasi person | 257K+ di full dataset |
| IoU threshold TP | 0.5 |
| Confidence threshold | 0.25 |
        """
    )
