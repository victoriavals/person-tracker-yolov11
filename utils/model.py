import streamlit as st
from pathlib import Path
from ultralytics import YOLO

# Path ke model fine-tuned terbaik (relatif dari root repo)
_FINETUNED_PATH = Path("runs/detect/runs/tune/strategy_A/weights/best.pt")

MODEL_OPTIONS = {
    "✅ Custom Fine-tuned (Strategy A — mAP@0.5: 0.739)": str(_FINETUNED_PATH),
    "yolo11n (Nano — Pretrained COCO, Cepat)": "yolo11n",
    "yolo11s (Small — Pretrained COCO, Seimbang)": "yolo11s",
}

PERSON_CLASS = 0  # COCO class index for 'person'


@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_path: str) -> YOLO:
    """
    Load YOLO model dengan caching.
    Jika model_path adalah path file lokal, load langsung.
    Jika nama model (yolo11n, dll), Ultralytics akan download otomatis.
    """
    model = YOLO(model_path)
    return model
