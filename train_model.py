import cv2
import os
import numpy as np
from PIL import Image
import pickle

dataset_path = "dataset"
trainer_path = "trainer/trainer.yml"
label_map_path = "trainer/labels.pkl"

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
face_samples = []
face_labels = []

for root, dirs, files in os.walk(dataset_path):
    for dir_name in dirs:
        person_path = os.path.join(root, dir_name)

        if dir_name not in label_ids:
            label_ids[dir_name] = current_id
            current_id += 1

        label = label_ids[dir_name]

        for image_name in os.listdir(person_path):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(person_path, image_name)

                image = Image.open(image_path).convert("L")
                image_np = np.array(image, "uint8")

                face_samples.append(image_np)
                face_labels.append(label)

print("Training model...")
recognizer.train(face_samples, np.array(face_labels))

recognizer.save(trainer_path)

with open(label_map_path, "wb") as f:
    pickle.dump(label_ids, f)

print("Model trained successfully.")
print("Label mapping:", label_ids)
