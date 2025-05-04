import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

def load_test_data(test_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

def evaluate_model(model_path, test_data):
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_data)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_directory = os.path.join('data', 'test')
    model_file_path = os.path.join('models', 'cnn_model.h5')
    
    test_data = load_test_data(test_directory)
    evaluate_model(model_file_path, test_data)