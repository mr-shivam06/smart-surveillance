import cv2
import pickle
import time
import face_recognition
import numpy as np

# Load database
with open("ai/face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

names = list(face_db.keys())
encodings = list(face_db.values())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
frame_count = 0

last_name = "Unknown"
last_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces EVERY frame
    face_locations = face_recognition.face_locations(rgb, model="hog")

    # Recognize every 6 frames (performance + stability)
    if frame_count % 6 == 0 and face_locations:
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for enc, loc in zip(face_encs, face_locations):
            distances = face_recognition.face_distance(encodings, enc)

            if len(distances) > 0 and np.min(distances) < 0.65:
                idx = np.argmin(distances)
                last_name = names[idx]
            else:
                last_name = "Unknown"

            last_box = loc

    # Draw cached result (NO FLICKER)
    if last_box:
        top, right, bottom, left = last_box
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(
            frame,
            last_name,
            (left, top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )

    cv2.imshow("Stable Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
