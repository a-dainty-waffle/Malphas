# database.py
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

def init_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def build_database(known_faces_dir: str) -> dict:
    """
    Returns {name: embedding_vector} for each image in known_faces.
    Filename (without extension) = person's name.
    """
    app = init_model()
    db = {}

    for filename in os.listdir(known_faces_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name = os.path.splitext(filename)[0]
        path = os.path.join(known_faces_dir, filename)

        img = cv2.imread(path)
        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"[WARN] No face found in {filename}")
            continue

        # take the largest face
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        db[name] = face.embedding

    return db


def save_database(db: dict, path: str = "face_db.pkl"):
    with open(path, "wb") as f:
        pickle.dump(db, f)


def load_database(path: str = "face_db.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)