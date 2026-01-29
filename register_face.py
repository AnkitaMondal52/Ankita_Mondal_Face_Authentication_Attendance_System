import cv2
import os
import time

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

user_name = input("Enter user name: ")

dataset_path = "dataset"
user_path = os.path.join(dataset_path, user_name)
os.makedirs(user_path, exist_ok=True)

# Use DirectShow backend for Windows stability
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Camera warm-up
time.sleep(2)

# Explicit window creation
cv2.namedWindow("Face Registration", cv2.WINDOW_NORMAL)

# Force window to foreground (Windows fix)
cv2.moveWindow("Face Registration", 100, 100)
cv2.setWindowProperty("Face Registration", cv2.WND_PROP_TOPMOST, 1)

count = 0
last_capture_time = 0
CAPTURE_INTERVAL = CAPTURE_INTERVAL = 0.5   # ~5 images per second
TOTAL_IMAGES = 30

print("Camera started.")
print("Keep your face inside the box.")
print("Capturing images automatically...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        # Capture images slowly
        if count < TOTAL_IMAGES and current_time - last_capture_time >= CAPTURE_INTERVAL:
            count += 1
            last_capture_time = current_time
            cv2.imwrite(f"{user_path}/{count}.jpg", face)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"Captured: {count}/{TOTAL_IMAGES}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Always show window
    cv2.imshow("Face Registration", frame)
    cv2.waitKey(1)

    # Release topmost after first render
    cv2.setWindowProperty("Face Registration", cv2.WND_PROP_TOPMOST, 0)

    # AUTO-CLOSE after 30 images
    if count >= TOTAL_IMAGES:
        print("Face registration completed successfully.")
        time.sleep(2)  # grace period so user can see final frame
        break

cap.release()
cv2.destroyAllWindows()

