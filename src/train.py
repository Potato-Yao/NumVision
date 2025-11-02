"""
Training script for the digit recognition model.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import DigitRecognitionModel


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


def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot training and validation accuracy/loss over epochs.

    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    history_dict = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history_dict['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(history_dict['loss'], label='Training Loss', marker='o')
    ax2.plot(history_dict['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")
    plt.close()


def visualize_predictions(model, x_test, y_test, num_samples=10,
                         save_path='models/predictions_sample.png'):
    """
    Visualize sample predictions from the test set.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[indices]
    sample_labels = y_test[indices]

    # Make predictions
    predictions, probabilities = model.predict(sample_images)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i, (img, true_label, pred_label, probs) in enumerate(
        zip(sample_images, sample_labels, predictions, probabilities)
    ):
        axes[i].imshow(img.squeeze(), cmap='gray')

        # Color based on correctness
        color = 'green' if pred_label == true_label else 'red'
        confidence = probs[pred_label] * 100

        axes[i].set_title(
            f'True: {true_label}, Pred: {pred_label}\n'
            f'Confidence: {confidence:.1f}%',
            color=color,
            fontsize=10
        )
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction samples saved to {save_path}")
    plt.close()


def train_model(epochs=10, batch_size=128, learning_rate=0.001):
    """
    Main training function.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate

    Returns:
        Trained model and test accuracy
    """
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

    # Plot training history
    plot_training_history(history)

    # Visualize predictions
    visualize_predictions(digit_model, x_test, y_test)

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

