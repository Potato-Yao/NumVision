"""
Main entry point for the NumVision digit recognition system.
"""
import sys
import os
import argparse
import numpy as np
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import DigitRecognitionModel
from src.train import train_model, load_and_preprocess_data
from src.predict import predict_digit, load_trained_model, batch_predict
from src.utils import (
    create_project_directories,
    plot_sample_images,
    save_test_results,
    get_model_info
)


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║              NumVision - Digit Recognition            ║
    ║           Handwritten Digit Recognition System        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def train_mode(args):
    """Train a new model."""
    print("\n🚀 Starting training mode...\n")

    model, accuracy = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    print(f"\n✅ Training completed with {accuracy*100:.2f}% test accuracy!")


def evaluate_mode(args):
    """Evaluate an existing model."""
    print("\n📊 Starting evaluation mode...\n")

    # Load model
    model = load_trained_model(args.model_path)
    if model is None:
        print("❌ Failed to load model. Please train a model first.")
        return

    # Load test data
    print("Loading test data...")
    _, _, (x_test, y_test) = load_and_preprocess_data()

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f"\n{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # Generate detailed results
    if args.detailed:
        print("\nGenerating detailed evaluation results...")
        save_test_results(model, x_test, y_test)
        print("✅ Detailed results saved to 'results/' directory")


def predict_mode(args):
    """Make predictions on images."""
    print("\n🔮 Starting prediction mode...\n")

    # Load model
    model = load_trained_model(args.model_path)
    if model is None:
        print("❌ Failed to load model. Please train a model first.")
        return

    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return

        print(f"Predicting digit from: {args.image}")
        predicted_digit, confidence, _ = predict_digit(model, args.image, show_plot=True)

        print(f"\n{'='*60}")
        print(f"Predicted Digit: {predicted_digit}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"{'='*60}")

    elif args.batch:
        # Batch prediction
        image_files = args.batch.split(',')
        print(f"Processing {len(image_files)} images...\n")

        results = batch_predict(model, image_files)

        print(f"\n{'='*60}")
        print("Batch Prediction Results:")
        print(f"{'='*60}")
        for result in results:
            if 'error' not in result:
                print(f"{result['image_path']}: Digit {result['predicted_digit']} "
                      f"({result['confidence']*100:.2f}%)")
            else:
                print(f"{result['image_path']}: Error - {result['error']}")

    else:
        print("❌ Please provide --image or --batch argument for prediction")


def interactive_mode(args):
    """Interactive mode for exploring the dataset and model."""
    print("\n🎮 Starting interactive mode...\n")

    # Load model
    model = load_trained_model(args.model_path)
    if model is None:
        print("⚠️ No trained model found. Loading dataset only.")
        model = None
    else:
        print("✅ Model loaded successfully!")
        get_model_info(model)

    # Load data
    print("\nLoading dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    print("\n" + "="*60)
    print("Interactive Mode - Available Commands:")
    print("="*60)
    print("1. View random samples from dataset")
    print("2. View model information")
    print("3. Make predictions on test samples")
    print("4. Evaluate model on test set")
    print("5. Exit")
    print("="*60)

    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                print("\nDisplaying random samples from training set...")
                plot_sample_images(x_train, y_train, num_samples=25,
                                 title='Random Training Samples')

            elif choice == '2':
                if model:
                    get_model_info(model)
                else:
                    print("❌ No model loaded")

            elif choice == '3':
                if model:
                    num_samples = int(input("How many samples to predict? (1-100): "))
                    num_samples = min(max(1, num_samples), 100)

                    indices = np.random.choice(len(x_test), num_samples, replace=False)
                    predictions = model.predict(x_test[indices], verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)

                    correct = np.sum(predicted_classes == y_test[indices])
                    print(f"\nPredictions: {correct}/{num_samples} correct "
                          f"({correct/num_samples*100:.2f}%)")
                else:
                    print("❌ No model loaded")

            elif choice == '4':
                if model:
                    print("\nEvaluating model on full test set...")
                    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
                    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
                else:
                    print("❌ No model loaded")

            elif choice == '5':
                print("\n👋 Exiting interactive mode...")
                break

            else:
                print("❌ Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def info_mode():
    """Display project information."""
    print("\n📖 NumVision - Project Information\n")
    print("="*60)
    print("Description:")
    print("  A handwritten digit recognition system using CNNs")
    print("  Built with TensorFlow/Keras on the MNIST dataset")
    print("\nModel Architecture:")
    print("  - 3 Convolutional layers (32, 64, 64 filters)")
    print("  - MaxPooling layers")
    print("  - Dense layer (128 units)")
    print("  - Dropout (0.5)")
    print("  - Softmax output (10 classes)")
    print("\nExpected Performance:")
    print("  - Training Accuracy: ~99%")
    print("  - Test Accuracy: ~98.5%")
    print("  - Training Time: ~5-10 minutes on CPU")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NumVision - Handwritten Digit Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a new model:
    python main.py --mode train --epochs 10 --batch-size 128
  
  Evaluate existing model:
    python main.py --mode evaluate --detailed
  
  Predict single image:
    python main.py --mode predict --image tests/digit.png
  
  Predict multiple images:
    python main.py --mode predict --batch "img1.png,img2.png,img3.png"
  
  Interactive mode:
    python main.py --mode interactive
        """
    )

    # Main arguments
    parser.add_argument('--mode', type=str,
                       choices=['train', 'evaluate', 'predict', 'interactive', 'info'],
                       default='info',
                       help='Operation mode')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')

    # Model arguments
    parser.add_argument('--model-path', type=str,
                       default='models/digit_recognition_model.h5',
                       help='Path to model file')

    # Prediction arguments
    parser.add_argument('--image', type=str,
                       help='Path to image file for prediction')
    parser.add_argument('--batch', type=str,
                       help='Comma-separated list of image paths')

    # Evaluation arguments
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed evaluation results')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Create necessary directories
    create_project_directories()

    # Execute based on mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'interactive':
        interactive_mode(args)
    elif args.mode == 'info':
        info_mode()

    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()

