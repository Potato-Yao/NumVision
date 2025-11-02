"""
Prediction utilities for digit recognition.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras


def preprocess_image(image_path, target_size=(28, 28)):
    """
    Load and preprocess an image for prediction.

    Args:
        image_path: Path to the image file
        target_size: Target size for the image

    Returns:
        Preprocessed image array
    """
    # Load image
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert('L')

    # Resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Invert if needed (MNIST has white digits on black background)
    # If the image has black digits on white background, invert it
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_digit(model, image_path, show_plot=True):
    """
    Predict a digit from an image file.

    Args:
        model: Trained model
        image_path: Path to the image file
        show_plot: Whether to display the prediction

    Returns:
        Predicted digit and confidence
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit]

    # Get all probabilities
    all_probs = predictions[0]

    if show_plot:
        visualize_prediction(
            processed_image[0].squeeze(),
            predicted_digit,
            confidence,
            all_probs
        )

    return predicted_digit, confidence, all_probs


def visualize_prediction(image, predicted_digit, confidence, probabilities):
    """
    Visualize the prediction with probability distribution.

    Args:
        image: Input image (28x28)
        predicted_digit: Predicted digit
        confidence: Confidence score
        probabilities: Probability distribution over all classes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Display the image
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Predicted Digit: {predicted_digit}\n'
                  f'Confidence: {confidence*100:.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Display probability distribution
    digits = np.arange(10)
    colors = ['green' if i == predicted_digit else 'steelblue' for i in digits]

    ax2.bar(digits, probabilities * 100, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Digit', fontsize=12)
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xticks(digits)
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, prob in enumerate(probabilities):
        if prob > 0.01:  # Only show labels for probabilities > 1%
            ax2.text(i, prob*100 + 1, f'{prob*100:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def batch_predict(model, image_paths):
    """
    Make predictions on multiple images.

    Args:
        model: Trained model
        image_paths: List of image paths

    Returns:
        List of predictions and confidences
    """
    results = []

    for img_path in image_paths:
        try:
            predicted_digit, confidence, _ = predict_digit(
                model, img_path, show_plot=False
            )
            results.append({
                'image_path': img_path,
                'predicted_digit': int(predicted_digit),
                'confidence': float(confidence)
            })
            print(f"{img_path}: Digit {predicted_digit} (Confidence: {confidence*100:.2f}%)")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            results.append({
                'image_path': img_path,
                'error': str(e)
            })

    return results


def predict_from_array(model, image_array):
    """
    Predict digit from a numpy array.

    Args:
        model: Trained model
        image_array: Image as numpy array (28x28 or 28x28x1)

    Returns:
        Predicted digit and confidence
    """
    # Ensure correct shape
    if image_array.shape == (28, 28):
        image_array = np.expand_dims(image_array, axis=-1)

    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)

    # Normalize if needed
    if image_array.max() > 1.0:
        image_array = image_array.astype('float32') / 255.0

    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit]

    return predicted_digit, confidence


def load_trained_model(model_path='models/digit_recognition_model.h5'):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    model_path = 'models/digit_recognition_model.h5'

    # Load model
    model = load_trained_model(model_path)

    if model is not None:
        # Example prediction (you would need to provide an image path)
        print("\nModel ready for predictions!")
        print("Use predict_digit(model, 'path/to/image.png') to make predictions")

