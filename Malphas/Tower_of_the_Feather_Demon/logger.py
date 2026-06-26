import cv2
import os
from datetime import datetime


def log_entry(frame, bbox, name: str, base_dir: str = "logs"):

    # InsightFace bbox:
    # [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, bbox)

    # Clamp to image bounds
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "image_path": None
        }

    face_crop = frame[y1:y2, x1:x2].copy()

    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    folder = os.path.join(
        base_dir,
        "identified" if name != "Unknown" else "unidentified"
    )

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, f"{name}_{ts_str}.jpg")

    cv2.imwrite(filepath, face_crop)

    return {
        "name": name,
        "timestamp": timestamp.isoformat(),
        "image_path": filepath
    }