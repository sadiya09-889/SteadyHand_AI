import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(28, 28, 1), num_classes=62):
    """
    Builds a lightweight Convolutional Neural Network (CNN) model for character recognition.
    Optimized for extremely fast CPU training while maintaining good accuracy.
    """
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
