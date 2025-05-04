import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def preprocess_image(image, target_size=(128, 128)):
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

def preprocess_dataset(folder):
    images = load_images_from_folder(folder)
    processed_images = [preprocess_image(img) for img in images]
    return np.array(processed_images)