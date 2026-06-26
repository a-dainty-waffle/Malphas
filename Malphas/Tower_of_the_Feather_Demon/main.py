from database import build_database, save_database
from detector import run


def get_source():
    print("\n[MALPHAS MODE SELECT]")
    print("1 - Webcam")
    print("2 - Video file\n")

    choice = input("Select mode: ")

    if choice == "2":
        path = input("Enter video path: ")
        return path

    return 0


# --------------------------
# BUILD DATABASE (UNCHANGED)
# --------------------------
db = build_database("known_faces")
save_database(db)

print(f"Loaded {len(db)} faces into database")


# --------------------------
# SELECT INPUT SOURCE
# --------------------------
source = get_source()

# --------------------------
# RUN DETECTOR
# --------------------------
run(video_source=source)