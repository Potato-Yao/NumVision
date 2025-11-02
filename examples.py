"""
Example usage script demonstrating how to use NumVision in your own code.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def example_1_simple_training():
    """Example 1: Simple model training."""
    print("\n" + "="*70)
    print("Example 1: Simple Model Training")
    print("="*70 + "\n")

    from src.model import DigitRecognitionModel
    from src.train import load_and_preprocess_data

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # Create and build model
    model = DigitRecognitionModel()
    model.build_model()
    model.compile_model(learning_rate=0.001)

    # Train
    model.train(x_train, y_train, x_val, y_val, epochs=3, batch_size=128)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nFinal Accuracy: {test_accuracy*100:.2f}%")


def example_2_load_and_predict():
    """Example 2: Load model and make predictions."""
    print("\n" + "="*70)
    print("Example 2: Load Model and Predict")
    print("="*70 + "\n")

    from src.predict import load_trained_model
    from src.train import load_and_preprocess_data
    import numpy as np

    # Load model
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("No trained model found. Train a model first!")
        return

    # Load test data
    _, _, (x_test, y_test) = load_and_preprocess_data()

    # Predict on a single image
    sample_idx = 42
    sample_image = x_test[sample_idx:sample_idx+1]
    true_label = y_test[sample_idx]

    predictions = model.predict(sample_image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit]

    print(f"True Label: {true_label}")
    print(f"Predicted: {predicted_digit}")
    print(f"Confidence: {confidence*100:.2f}%")


def example_3_predict_from_file():
    """Example 3: Predict digit from image file."""
    print("\n" + "="*70)
    print("Example 3: Predict from Image File")
    print("="*70 + "\n")

    from src.predict import load_trained_model, predict_digit
    import os

    # Load model
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("No trained model found. Train a model first!")
        return

    # Check if test images exist
    test_image = 'tests/digit_5_simple.png'

    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("Create test images first: python create_test_images.py")
        return

    # Predict
    predicted_digit, confidence, all_probs = predict_digit(
        model, test_image, show_plot=False
    )

    print(f"Image: {test_image}")
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll probabilities:")
    for digit, prob in enumerate(all_probs):
        print(f"  Digit {digit}: {prob*100:.2f}%")


def example_4_custom_architecture():
    """Example 4: Create a custom model architecture."""
    print("\n" + "="*70)
    print("Example 4: Custom Model Architecture")
    print("="*70 + "\n")

    from tensorflow.keras import layers, models

    # Build a simpler model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                     input_shape=(28, 28, 1), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Custom model created!")
    model.summary()


def example_5_batch_prediction():
    """Example 5: Batch prediction on multiple images."""
    print("\n" + "="*70)
    print("Example 5: Batch Prediction")
    print("="*70 + "\n")

    from src.predict import load_trained_model, batch_predict
    import os
    import glob

    # Load model
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("No trained model found. Train a model first!")
        return

    # Get all test images
    test_images = glob.glob('tests/digit_*_simple.png')

    if not test_images:
        print("No test images found. Create them first:")
        print("  python create_test_images.py")
        return

    print(f"Found {len(test_images)} test images\n")

    # Predict
    results = batch_predict(model, test_images[:5])  # First 5 images

    print("\nBatch prediction complete!")


def example_6_save_and_load():
    """Example 6: Save and load models."""
    print("\n" + "="*70)
    print("Example 6: Save and Load Models")
    print("="*70 + "\n")

    from src.model import DigitRecognitionModel
    import os

    # Create a simple model
    model = DigitRecognitionModel()
    model.build_model()
    model.compile_model()

    # Save
    save_path = 'models/example_model.h5'
    model.save_model(save_path)
    print(f"✅ Model saved to: {save_path}")

    # Load
    loaded_model = DigitRecognitionModel()
    loaded_model.load_model(save_path)
    print(f"✅ Model loaded from: {save_path}")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"✅ Cleaned up example model")


def example_7_evaluation_metrics():
    """Example 7: Get detailed evaluation metrics."""
    print("\n" + "="*70)
    print("Example 7: Detailed Evaluation Metrics")
    print("="*70 + "\n")

    from src.predict import load_trained_model
    from src.train import load_and_preprocess_data
    from src.utils import (
        plot_confusion_matrix,
        generate_classification_report,
        analyze_misclassifications
    )

    # Load model
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("No trained model found. Train a model first!")
        return

    # Load test data
    _, _, (x_test, y_test) = load_and_preprocess_data()

    # Generate metrics
    print("Generating confusion matrix...")
    plot_confusion_matrix(model, x_test, y_test)

    print("\nGenerating classification report...")
    generate_classification_report(model, x_test, y_test)

    print("\nAnalyzing misclassifications...")
    analyze_misclassifications(model, x_test, y_test)


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              NumVision - Example Usage Scripts               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("\nAvailable Examples:")
    print("  1. Simple model training")
    print("  2. Load model and predict")
    print("  3. Predict from image file")
    print("  4. Custom model architecture")
    print("  5. Batch prediction")
    print("  6. Save and load models")
    print("  7. Detailed evaluation metrics")
    print("  8. Run all examples")
    print("  0. Exit")

    while True:
        try:
            choice = input("\nSelect an example (0-8): ").strip()

            if choice == '0':
                print("\nGoodbye!")
                break
            elif choice == '1':
                example_1_simple_training()
            elif choice == '2':
                example_2_load_and_predict()
            elif choice == '3':
                example_3_predict_from_file()
            elif choice == '4':
                example_4_custom_architecture()
            elif choice == '5':
                example_5_batch_prediction()
            elif choice == '6':
                example_6_save_and_load()
            elif choice == '7':
                example_7_evaluation_metrics()
            elif choice == '8':
                print("\nRunning all examples...")
                example_4_custom_architecture()
                example_6_save_and_load()
                example_2_load_and_predict()
                example_3_predict_from_file()
                example_5_batch_prediction()
                print("\n✅ All examples completed!")
            else:
                print("Invalid choice. Please enter 0-8.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

