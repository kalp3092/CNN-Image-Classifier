import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn_model import CNNModel

def load_data(train_dir, test_dir, img_size=(128, 128), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

def train_model(train_generator, test_generator, epochs=10):
    model = CNNModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=epochs, validation_data=test_generator)

    return model

def save_model(model, model_path):
    model.save(model_path)

if __name__ == "__main__":
    train_dir = os.path.join('data', 'train')
    test_dir = os.path.join('data', 'test')
    model_path = 'cnn_model.h5'

    train_generator, test_generator = load_data(train_dir, test_dir)
    model = train_model(train_generator, test_generator, epochs=10)
    save_model(model, model_path)