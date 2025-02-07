# models/face_recognition.py
import cv2
import numpy as np

def process_faces(image_path):
    face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return "No faces detected!"
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    output_path = image_path.replace('uploads', 'processed')
    cv2.imwrite(output_path, img)
    return f"Face recognition completed. Saved to {output_path}"
