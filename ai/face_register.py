import os
import pickle
import face_recognition
import numpy as np

KNOWN_DIR = "ai/known_faces"
db = {}

for file in os.listdir(KNOWN_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        name = file.split("_")[0]  # shivam_1 → shivam
        path = os.path.join(KNOWN_DIR, file)

        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)

        if encs:
            if name not in db:
                db[name] = []
            db[name].append(encs[0])

# Average encodings (KEY FIX)
final_db = {}
for name, enc_list in db.items():
    final_db[name] = np.mean(enc_list, axis=0)

with open("ai/face_db.pkl", "wb") as f:
    pickle.dump(final_db, f)

print("Registered identities:", list(final_db.keys()))
