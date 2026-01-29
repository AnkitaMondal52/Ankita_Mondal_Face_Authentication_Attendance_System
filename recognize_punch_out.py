import cv2
import pickle
import csv
import time
from datetime import datetime

# ---------------- Load Face Detector ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- Load Model ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

with open("trainer/labels.pkl", "rb") as f:
    label_ids = pickle.load(f)

labels = {v: k for k, v in label_ids.items()}

attendance_file = "attendance.csv"

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1)  # camera warm-up

# Window foreground fix (Windows)
cv2.namedWindow("Punch Out", cv2.WINDOW_NORMAL)
cv2.moveWindow("Punch Out", 100, 100)
cv2.setWindowProperty("Punch Out", cv2.WND_PROP_TOPMOST, 1)

print("PUNCH OUT MODE")

with open(attendance_file, "r") as f:
    rows = list(csv.reader(f))

# 🔹 UI tuning
PADDING = 30          # Bigger face box
DISPLAY_DELAY = 4000  # 4 seconds on screen after recognition

# 🔹 Recognition control (slow it down slightly)
FRAME_SKIP = 2              # process every 2nd frame
RECOGNITION_DELAY = 1.2     # seconds
last_recognition_time = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face)

        # 🔹 Bigger face box
        x1 = max(0, x - PADDING)
        y1 = max(0, y - PADDING)
        x2 = min(frame.shape[1], x + w + PADDING)
        y2 = min(frame.shape[0], y + h + PADDING)

        if confidence < 65 and label in labels:
            name = labels[label]
            time_out = datetime.now().strftime("%H:%M:%S")

            for row in reversed(rows):
                if row[0] == name and row[3] == "":
                    row[3] = time_out
                    break

            with open(attendance_file, "w", newline="") as f:
                csv.writer(f).writerows(rows)

            # Draw final box + text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"{name} - Punch Out",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("Punch Out", frame)
            cv2.waitKey(DISPLAY_DELAY)  # 🔹 stay visible longer

            print(f"{name} punched OUT at {time_out}")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Normal preview box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            "Punch Out",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Punch Out", frame)
    cv2.waitKey(1)

    # Release topmost after first render
    cv2.setWindowProperty("Punch Out", cv2.WND_PROP_TOPMOST, 0)

cap.release()
cv2.destroyAllWindows()
