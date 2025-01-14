import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from numpy import linalg as LA

# load the pre-trained FaceNet model
model = load_model('facenet_keras.h5')
print("Model Loaded Successfully")


# image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img

def get_face_embedding(model, image_path):

    img = preprocess_image(image_path)

    embedding = model.predict(img)

    return embedding

def compare_faces(embedding1, embedding2, threshold=0.5):

    distance = LA.norm(embedding1 - embedding2)

    if distance < threshold:
        print("Face Matched")
    else:
        print("Faces are different.")

    return distance

embedding1 = get_face_embedding("<images>")

embedding2 = get_face_embedding("<images>")

distance = compare_faces(embedding1, embedding2)

print(f"Euclidean Distance: {distance}")


