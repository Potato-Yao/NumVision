"""
Quick start script - trains a model with default settings.
Perfect for first-time users.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_model


def main():
    """Quick start training with optimal default settings."""

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           NumVision - Quick Start Training                    ║
║                                                               ║
║  This script will train a digit recognition model with       ║
║  default settings. It should take about 5-10 minutes.        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    print("\nTraining Configuration:")
    print("  • Epochs: 10")
    print("  • Batch Size: 128")
    print("  • Learning Rate: 0.001")
    print("  • Dataset: MNIST (60k training, 10k test)")
    print("  • Model: Custom CNN (3 conv layers + 2 dense layers)")
    print()

    response = input("Start training? (y/n): ").strip().lower()

    if response != 'y':
        print("Training cancelled.")
        return

    print("\n🚀 Starting training...\n")
    print("="*70)

    try:
        # Train the model
        model, accuracy = train_model(
            epochs=10,
            batch_size=128,
            learning_rate=0.001
        )

        print("\n" + "="*70)
        print("🎉 Training Complete!")
        print("="*70)
        print(f"\n✅ Final Test Accuracy: {accuracy*100:.2f}%")
        print(f"✅ Model saved to: models/digit_recognition_model.h5")
        print(f"✅ Training history plot: models/training_history.png")
        print(f"✅ Sample predictions: models/predictions_sample.png")

        print("\n📋 Next Steps:")
        print("  1. Evaluate the model:")
        print("     python main.py --mode evaluate --detailed")
        print()
        print("  2. Test with your own images:")
        print("     python main.py --mode predict --image your_image.png")
        print()
        print("  3. Use the drawing interface:")
        print("     python src/draw_interface.py")
        print()
        print("  4. Interactive exploration:")
        print("     python main.py --mode interactive")
        print()

        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

