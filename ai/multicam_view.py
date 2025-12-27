import cv2
import time
import pickle
import numpy as np
from ultralytics import YOLO
import face_recognition

# ================= LOAD MODELS =================
yolo = YOLO("yolov8n.pt")

with open("ai/face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

face_names = list(face_db.keys())
face_encs = list(face_db.values())

# ================= CAMERA SOURCES =================
camera_sources = [
    "videos/sample1.mp4",              # CCTV video
    0,                                 # Laptop webcam
    "http://100.78.72.145:4747/video"    # DroidCam (CHANGE IP)
]

caps = [cv2.VideoCapture(src) for src in camera_sources]

# 🔒 FIXED DISPLAY SIZE (IMPORTANT)
FRAME_W, FRAME_H = 640, 480

# ================= CONTROL =================
prev_time = 0
frame_id = 0
cached_faces = []

# ================= MAIN LOOP =================
while True:
    frames = []

    for cap in caps:
        ret, frame = cap.read()

        # Loop video files
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        else:
            frame = None

        frames.append(frame)

    if frames[0] is None:
        break

    frame_id += 1

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    output_frames = []

    for frame in frames:
        if frame is None:
            output_frames.append(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
            continue

        # ---------- YOLO DETECTION ----------
        results = yolo(frame, conf=0.4, verbose=False)[0]

        # ---------- FACE RECOGNITION (EVERY 12 FRAMES) ----------
        if frame_id % 12 == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb_small, model="hog")
            encs = face_recognition.face_encodings(rgb_small, locs)

            cached_faces = []
            for enc, (t, r, b, l) in zip(encs, locs):
                dists = face_recognition.face_distance(face_encs, enc)
                name = "Unknown"
                if len(dists) > 0 and np.min(dists) < 0.65:
                    name = face_names[np.argmin(dists)]
                cached_faces.append((t*2, r*2, b*2, l*2, name))

        # ---------- DRAW OBJECTS ----------
        for box in results.boxes:
            cls = int(box.cls[0])
            label = yolo.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            text = label
            if label == "person":
                for (t, r, b, l, name) in cached_faces:
                    cx, cy = (l+r)//2, (t+b)//2
                    if x1 < cx < x2 and y1 < cy < y2:
                        text = f"person ({name})"
                        break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        output_frames.append(frame)

    # ---------- SAFE STACK ----------
    combined = np.hstack(output_frames)

    cv2.putText(combined, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Multi-Camera Surveillance (ALL SOURCES)", combined)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
