'''import cv2              # OpenCV library for computer vision tasks
import os               # OS library to create folders and manage file paths

# Load the Haarcascade XML file used for face detection
# This pretrained model detects frontal faces in images
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the folder where captured face images will be stored
person_name=input("Enter the name of the person ")
dataset_path = "dataset/"+person_name

# Create the folder if it does not already exist
# exist_ok=True prevents error if folder already exists
os.makedirs(dataset_path, exist_ok=True)

# Start webcam capture
# 0 means default system camera
cap = cv2.VideoCapture(0)

# Counter to number saved face images
count = 0

# Infinite loop to continuously capture frames from webcam

while True:

    # Read a frame from webcam
    # ret = True if frame successfully captured
    # frame = the actual image captured
    ret, frame = cap.read()

    # Convert the captured frame to grayscale
    # Haarcascade works faster on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    # detectMultiScale parameters:
    # gray → input image
    # 1.3 → scale factor (image reduction per iteration)
    # 5 → minimum neighbors required for detection
    faces = face_cascade.detectMultiScale(image=gray,
                                          scaleFactor=1.3,
                                           minNeighbors=5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:

        # Crop only the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face image to 224x224
        # Required input size for VGG16 model
        face = cv2.resize(face, (224,224))

        # Create file name for saving image
        file_name = f"{dataset_path}/{count}.jpg"

        # Save the cropped face image to dataset folder
        cv2.imwrite(file_name, face)

        # Increase image counter
        count += 1

        # Draw rectangle around detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,str(count),(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)
    # Display the video frame with face rectangle
    cv2.imshow("Face Capture", frame)

    # Stop capturing if:
    # 1️⃣ ESC key pressed
    # 2️⃣ 100 images collected
    if cv2.waitKey(1) == 27 or count > 100:
        break
# Release webcam
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()'''


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
