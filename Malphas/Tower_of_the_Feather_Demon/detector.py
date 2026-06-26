import cv2
import numpy as np
import time
import json
import os
from collections import deque
from insightface.app import FaceAnalysis
from database import load_database
from logger import log_entry


# ---------------------------
# Utils
# ---------------------------

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def init_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def find_known_image(name):
    """Find reference image from Known_Faces folder"""
    folder = "Known_Faces"
    for ext in [".jpg", ".png", ".jpeg"]:
        path = os.path.join(folder, name + ext)
        if os.path.exists(path):
            return path
    return None


# ---------------------------
# ENGINE
# ---------------------------

def run(video_source=0, threshold=0.6, check_every_n_frames=5):

    db = load_database()
    known_names = list(db.keys())
    known_embeddings = list(db.values())

    app = init_model()

    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) if isinstance(
        video_source, int
    ) else cv2.VideoCapture(video_source)

    logs = []
    log_feed = deque(maxlen=10)

    seen_in_frame = {}
    last_seen_time = {}

    frame_counter = 0
    fps = 0
    fps_time = time.time()
    frame_id = 0

    face_data = []

    # ---------------------------
    # MAIN LOOP
    # ---------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_counter += 1

        # FPS
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_counter
            frame_counter = 0
            fps_time = now

        run_ai = (frame_id % check_every_n_frames == 0)

        if run_ai:
            faces = app.get(frame)

            face_data = []
            current_frame_names = set()

            for face in faces:
                embedding = face.embedding
                bbox = face.bbox.astype(int)

                name = "Unknown"
                best_score = -1

                for i, known_emb in enumerate(known_embeddings):
                    score = cosine_similarity(embedding, known_emb)
                    if score > best_score:
                        best_score = score
                        name = known_names[i]

                if best_score < threshold:
                    name = "Unknown"

                current_frame_names.add(name)
                last_seen_time[name] = time.time()

                # ---------------------------
                # Reference image
                # ---------------------------
                known_path = find_known_image(name) if name != "Unknown" else None

                # ---------------------------
                # Log only first appearance
                # ---------------------------
                entry = None

                if not seen_in_frame.get(name, False):

                    entry = log_entry(frame, bbox, name)

                    logs.append(entry)

                    log_feed.appendleft({
                        "name": name,
                        "timestamp": entry["timestamp"],
                        "known_face": known_path,
                        "detected_face": entry["image_path"]
                    })

                    print(f"[LOG] {entry['timestamp']} {name} ({best_score:.2f})")

                seen_in_frame[name] = True

                # ---------------------------
                # Store for rendering
                # ---------------------------
                face_data.append({
                    "bbox": bbox,
                    "name": name,
                    "score": float(best_score)
                })
        # CLEANUP
        for person in list(seen_in_frame.keys()):
            if time.time() - last_seen_time.get(person, 0) > 2:
                seen_in_frame[person] = False

        # ---------------------------
        # RENDER
        # ---------------------------
        display = frame.copy()

        for item in face_data:
            x1, y1, x2, y2 = item["bbox"]
            name = item["name"]
            score = item["score"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                f"{name} ({score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # ---------------------------
        # OUTPUT
        # ---------------------------
        yield {
            "frame": display,
            "fps": fps,
            "logs": list(log_feed),
        }

    cap.release()

    with open("logs/log.json", "w") as f:
        json.dump(logs, f, indent=2)