

import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ask person name
person_name = input("Enter the name of the person: ")
dataset_path = os.path.join("dataset", person_name)

# Create folder
os.makedirs(dataset_path, exist_ok=True)

# Use DirectShow backend (IMPORTANT FIX for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Try camera index 1 if 0 fails
if not cap.isOpened():
    print("Camera 0 failed. Trying camera 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

count = 0

while True:
    ret, frame = cap.read()

    # If frame not captured, skip this loop
    if not ret or frame is None:
        print("Warning: Failed to grab frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

        file_name = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(file_name, face)

        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(count), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1)
    if key == 27 or count >= 50:   # ESC key or 100 images
        break

cap.release()
cv2.destroyAllWindows()
