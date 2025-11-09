"""
Main entry point for the NumVision digit recognition system.
"""
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_model
from src.utils import create_project_directories
from src.draw_interface import launch_drawing_app
from src.predict import load_trained_model


def train_mode(args):
    """Train a new model."""
    print("\n🚀 Starting training mode...\n")

    model, accuracy = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_gpu=args.use_gpu,
        mixed_precision=args.mixed_precision
    )

    print(f"\n✅ Training completed with {accuracy*100:.2f}% test accuracy!")


def gui_mode(args):
    """Launch the drawing GUI."""
    print("\n🎨 Launching drawing interface...\n")

    # Load model
    model = load_trained_model(args.model_path)
    if model is None:
        print("⚠️ No trained model found. You can still draw, but prediction won't work.")
        print("   Train a model first using: python main.py --mode train\n")

    launch_drawing_app(model)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NumVision - Handwritten Digit Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a new model:
    python main.py --mode train --epochs 10 --batch-size 128
  
  Launch drawing GUI:
    python main.py --mode gui
  
  Train with CPU only:
    python main.py --mode train --no-gpu
        """
    )

    # Main arguments
    parser.add_argument('--mode', type=str,
                       choices=['train', 'gui'],
                       default='gui',
                       help='Operation mode: train or gui (default: gui)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU for training if available (default: True)')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                       help='Disable GPU and use CPU only')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training for better GPU performance (default: True)')
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false',
                       help='Disable mixed precision training')

    # Model arguments
    parser.add_argument('--model-path', type=str,
                       default='models/digit_recognition_model.h5',
                       help='Path to model file')

    args = parser.parse_args()

    # Create necessary directories
    create_project_directories()

    # Execute based on mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'gui':
        gui_mode(args)

    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()

