# import required  libraries
import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import os

model = YOLO('yolov8n.pt')

known_faces_dir = 'known_faces'

known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'webp')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])

            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

cap = cv2.VideoCapture(0)

