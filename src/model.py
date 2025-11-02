"""
CNN Model Architecture for Handwritten Digit Recognition
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class DigitRecognitionModel:
    """
    Convolutional Neural Network for digit recognition.
    Custom implementation with training functionality.
    """

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initialize the model architecture.

        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (0-9 digits)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build the CNN architecture from scratch.

        Architecture:
        - Conv2D layer (32 filters, 3x3 kernel) + ReLU
        - MaxPooling2D (2x2)
        - Conv2D layer (64 filters, 3x3 kernel) + ReLU
        - MaxPooling2D (2x2)
        - Conv2D layer (64 filters, 3x3 kernel) + ReLU
        - Flatten
        - Dense (128 units) + ReLU + Dropout
        - Dense (10 units) + Softmax
        """
        self.model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=self.input_shape,
                         padding='same',
                         name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu',
                         padding='same',
                         name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),

            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu',
                         padding='same',
                         name='conv3'),

            # Flatten and Dense Layers
            layers.Flatten(name='flatten'),
            layers.Dense(128, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])

        print("Model architecture built successfully!")
        return self.model

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation. Call build_model() first.")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model compiled successfully!")

    def get_summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model must be built first. Call build_model() first.")

        return self.model.summary()

    def train(self, x_train, y_train, x_val, y_val,
              epochs=10, batch_size=128, verbose=1):
        """
        Train the model on provided data.

        Args:
            x_train: Training images
            y_train: Training labels
            x_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode (0, 1, or 2)

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training.")

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        # Model checkpoint callback
        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Batch size: {batch_size}\n")

        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, checkpoint, lr_scheduler],
            verbose=verbose
        )

        print("\nTraining completed!")
        return self.history

    def evaluate(self, x_test, y_test, verbose=1):
        """
        Evaluate model performance on test data.

        Args:
            x_test: Test images
            y_test: Test labels
            verbose: Verbosity mode

        Returns:
            Test loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation.")

        print("\nEvaluating model on test data...")
        test_loss, test_accuracy = self.model.evaluate(
            x_test, y_test, verbose=verbose
        )

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        return test_loss, test_accuracy

    def predict(self, images):
        """
        Make predictions on new images.

        Args:
            images: Array of images to predict

        Returns:
            Predicted class labels and probabilities
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction.")

        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes, predictions

    def save_model(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def get_training_history(self):
        """
        Get training history metrics.

        Returns:
            Dictionary containing training history
        """
        if self.history is None:
            raise ValueError("Model must be trained first.")

        return self.history.history

