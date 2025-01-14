import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

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



