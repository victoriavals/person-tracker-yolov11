import streamlit as st
from ultralytics import YOLO

MODEL_OPTIONS = {
    "yolo11n (Nano — Cepat, Aman di Cloud)": "yolo11n",
    "yolo11s (Small — Seimbang)": "yolo11s",
    "yolo11m (Medium — Akurat, Risiko RAM)": "yolo11m",
}

PERSON_CLASS = 0  # COCO class index for 'person'


@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_name: str) -> YOLO:
    """
    Load YOLO model with caching.
    Model di-download otomatis oleh ultralytics jika belum ada.
    Di-cache per session agar tidak reload saat interaksi lain.
    """
    model = YOLO(f"{model_name}.pt")
    return model
