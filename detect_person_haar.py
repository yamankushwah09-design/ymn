################################################################################
#                        NVIDIA Cloud Lab by AiProff.ai
#              Task 1: Haar Cascade Face Detection (Reference Code)
################################################################################

# Step 1: Import required libraries
import cv2
import os

# Step 2: Load image using cv2.imread() which reads image from file path and returns numpy array
img = cv2.imread("image.jpg")
if img is None:
    raise SystemExit("Failed to load 'image.jpg' — make sure file exists in the script folder.")

# Step 3: Convert image to grayscale using cv2.cvtColor() as Haar Cascade works better on grayscale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 4: Load pre-trained Haar Cascade classifier for face detection
# FIXED: Since the XML file is in the current directory (not in a subdirectory),
# we use just the filename without any folder path
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise SystemExit(f"Failed to load cascade at {cascade_path}")

# Step 5: Detect faces using detectMultiScale() with scaleFactor for pyramid scaling, minNeighbors for detection quality
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Step 6: Draw blue bounding boxes around detected faces using cv2.rectangle() with BGR color format (255,0,0)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Step 7: Display result in window using cv2.imshow() and wait for key press with cv2.waitKey()
# Note: Commented out because GUI might not be available on headless systems
# cv2.imshow("Haar Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 8: Save output image to disk using cv2.imwrite() and print face detection count
cv2.imwrite("task1_face_detected.jpg", img)
print(f"Detected {len(faces)} face(s). Output saved to face_detected.jpg")