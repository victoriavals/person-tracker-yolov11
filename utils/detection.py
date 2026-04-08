import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils.visualization import draw_detections
from utils.model import PERSON_CLASS


def detect_image(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Jalankan deteksi orang pada satu gambar.

    Args:
        model      : YOLO model yang sudah dimuat
        image      : numpy array BGR (dari cv2) atau RGB (dari PIL)
        conf       : confidence threshold
        iou        : IoU threshold untuk NMS

    Returns:
        annotated_bgr : numpy array BGR dengan bounding box ter-overlay
        stats         : dict dengan info deteksi
    """
    results = model.predict(
        source=image,
        classes=[PERSON_CLASS],
        conf=conf,
        iou=iou,
        verbose=False,
    )

    boxes, confs = [], []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls.item()) == PERSON_CLASS:
                boxes.append(box.xyxy[0].cpu().numpy())
                confs.append(float(box.conf.item()))

    annotated = draw_detections(image, boxes, confs)

    stats = {
        "persons_found": len(boxes),
        "avg_confidence": float(np.mean(confs)) if confs else 0.0,
        "max_confidence": float(np.max(confs)) if confs else 0.0,
        "min_confidence": float(np.min(confs)) if confs else 0.0,
        "confidences": confs,
    }

    return annotated, stats


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Konversi PIL Image (RGB) ke numpy BGR untuk OpenCV."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Konversi BGR ke RGB untuk Streamlit st.image()."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
