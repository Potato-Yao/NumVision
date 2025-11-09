"""
Training script for the digit recognition model.
"""
import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

from .model import DigitRecognitionModel
from .gpu_config import configure_for_training


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        Preprocessed training, validation, and test data
    """
    print("Loading MNIST dataset...")

    # Load MNIST dataset
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Original training data shape: {x_train_full.shape}")
    print(f"Original test data shape: {x_test.shape}")

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full
    )

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape to include channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"\nPreprocessed data shapes:")
    print(f"Training: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Validation: {x_val.shape}, Labels: {y_val.shape}")
    print(f"Test: {x_test.shape}, Labels: {y_test.shape}")

    # Display class distribution
    print(f"\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} samples")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)





def train_model(epochs=10, batch_size=128, learning_rate=0.001, use_gpu=True, mixed_precision=True):
    """
    Main training function.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        use_gpu: Whether to use GPU if available (default: True)
        mixed_precision: Enable mixed precision training for better GPU performance (default: True)

    Returns:
        Trained model and test accuracy
    """
    # Configure GPU settings
    if use_gpu:
        print("\n" + "="*60)
        print("Configuring GPU for Training")
        print("="*60)
        gpu_available = configure_for_training(
            use_mixed_precision=mixed_precision,
            memory_growth=True
        )

        # Adjust batch size for GPU if available
        if gpu_available and batch_size == 128:
            # GPU can typically handle larger batch sizes for better performance
            batch_size = 256
            print(f"Batch size increased to {batch_size} for GPU training")
    else:
        print("\nGPU training disabled. Using CPU...")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # Create and build model
    print("\n" + "="*60)
    print("Building CNN Model")
    print("="*60)

    digit_model = DigitRecognitionModel(input_shape=(28, 28, 1), num_classes=10)
    digit_model.build_model()
    digit_model.compile_model(learning_rate=learning_rate)

    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    digit_model.get_summary()

    # Train the model
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)

    history = digit_model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)

    test_loss, test_accuracy = digit_model.evaluate(x_test, y_test)

    # Save the final model
    model_path = 'models/digit_recognition_model.h5'
    digit_model.save_model(model_path)


    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Model saved to: {model_path}")
    print("="*60 + "\n")

    return digit_model, test_accuracy


if __name__ == "__main__":
    # Train with default parameters
    trained_model, accuracy = train_model(
        epochs=10,
        batch_size=128,
        learning_rate=0.001
    )

