"""
Complete demo script showcasing all features of NumVision.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_training():
    """Demonstrate model training."""
    print_section("DEMO 1: Training the Model")

    from src.train import train_model

    print("Training a digit recognition model...")
    print("This will take approximately 5-10 minutes.\n")

    model, accuracy = train_model(epochs=5, batch_size=128, learning_rate=0.001)

    print(f"\n✅ Training completed with {accuracy*100:.2f}% accuracy!")
    print("Model saved to: models/digit_recognition_model.h5")


def demo_evaluation():
    """Demonstrate model evaluation."""
    print_section("DEMO 2: Evaluating the Model")

    from src.train import load_and_preprocess_data
    from src.predict import load_trained_model
    from src.utils import save_test_results

    # Load model
    print("Loading trained model...")
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("❌ No trained model found. Please run demo_training() first.")
        return

    # Load test data
    print("Loading test data...")
    _, _, (x_test, y_test) = load_and_preprocess_data()

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")

    # Generate detailed results
    print("\nGenerating detailed results...")
    save_test_results(model, x_test, y_test)
    print("✅ Results saved to 'results/' directory")


def demo_prediction():
    """Demonstrate prediction on test images."""
    print_section("DEMO 3: Making Predictions")

    from src.predict import load_trained_model
    from src.train import load_and_preprocess_data
    import numpy as np

    # Load model
    print("Loading trained model...")
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("❌ No trained model found. Please run demo_training() first.")
        return

    # Load test data
    print("Loading test data...")
    _, _, (x_test, y_test) = load_and_preprocess_data()

    # Make predictions on random samples
    print("\nMaking predictions on random test samples...")
    num_samples = 5
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    for i, idx in enumerate(indices):
        img = x_test[idx:idx+1]
        true_label = y_test[idx]

        predictions = model.predict(img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]

        status = "✅" if predicted_digit == true_label else "❌"
        print(f"{status} Sample {i+1}: True={true_label}, "
              f"Predicted={predicted_digit}, Confidence={confidence*100:.2f}%")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print_section("DEMO 4: Data Visualization")

    from src.train import load_and_preprocess_data
    from src.utils import plot_sample_images

    print("Loading dataset...")
    (x_train, y_train), _, _ = load_and_preprocess_data()

    print("\nDisplaying sample images from training set...")
    print("(A matplotlib window will open)")

    plot_sample_images(x_train, y_train, num_samples=25,
                      title='MNIST Training Samples')

    print("✅ Visualization displayed!")


def demo_create_test_images():
    """Demonstrate creating test images."""
    print_section("DEMO 5: Creating Test Images")

    from create_test_images import create_simple_digit_image
    import os

    os.makedirs('tests', exist_ok=True)

    print("Creating sample digit images...")

    for digit in range(5):
        create_simple_digit_image(digit, save_path=f'tests/demo_digit_{digit}.png')

    print("\n✅ Sample images created in 'tests/' directory")


def demo_drawing_interface():
    """Demonstrate the drawing interface."""
    print_section("DEMO 6: Interactive Drawing Interface")

    from src.predict import load_trained_model
    from src.draw_interface import launch_drawing_app

    print("Loading trained model...")
    model = load_trained_model('models/digit_recognition_model.h5')

    if model is None:
        print("⚠️  No trained model found. Launching without prediction capability.")
        print("You can still use the interface to draw and save images.")

    print("\nLaunching drawing interface...")
    print("Draw a digit and click 'Predict' to see the result!")
    print("(A window will open)")

    launch_drawing_app(model)


def run_full_demo():
    """Run the complete demo."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              NumVision - Complete Demo                       ║
    ║          Handwritten Digit Recognition System                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print("\nThis demo will showcase all features of NumVision:")
    print("  1. Model Training")
    print("  2. Model Evaluation")
    print("  3. Making Predictions")
    print("  4. Data Visualization")
    print("  5. Creating Test Images")
    print("  6. Interactive Drawing Interface")

    print("\n⚠️  Note: The full demo will take 10-15 minutes.")
    response = input("\nRun full demo? (y/n): ").strip().lower()

    if response != 'y':
        print("\nDemo cancelled. You can run individual demos:")
        print("  python demo.py --quick")
        return

    try:
        # Run all demos
        demo_training()
        demo_evaluation()
        demo_prediction()
        demo_visualization()
        demo_create_test_images()

        print_section("🎉 Demo Complete!")
        print("All features demonstrated successfully!")
        print("\nTo launch the drawing interface:")
        print("  python src/draw_interface.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


def run_quick_demo():
    """Run a quick demo without training."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            NumVision - Quick Demo                            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print("\nQuick demo (requires pre-trained model):")

    try:
        demo_evaluation()
        demo_prediction()
        demo_create_test_images()

        print_section("✅ Quick Demo Complete!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease train a model first:")
        print("  python quickstart.py")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='NumVision Demo Script')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo (no training)')
    parser.add_argument('--demo', type=str,
                       choices=['train', 'eval', 'predict', 'viz', 'images', 'draw'],
                       help='Run specific demo')

    args = parser.parse_args()

    if args.quick:
        run_quick_demo()
    elif args.demo:
        if args.demo == 'train':
            demo_training()
        elif args.demo == 'eval':
            demo_evaluation()
        elif args.demo == 'predict':
            demo_prediction()
        elif args.demo == 'viz':
            demo_visualization()
        elif args.demo == 'images':
            demo_create_test_images()
        elif args.demo == 'draw':
            demo_drawing_interface()
    else:
        run_full_demo()


if __name__ == "__main__":
    main()

