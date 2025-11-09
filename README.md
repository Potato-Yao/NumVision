# NumVision - Digit Recognition System

A simplified handwritten digit recognition system using CNN and MNIST dataset.

## Features

- **Model Training**: Train a CNN model on MNIST dataset
- **GUI Interface**: Draw digits and get real-time predictions

## Quick Start

### 1. Train the Model

```bash
python main.py --mode train --epochs 10
```

This will:
- Download MNIST dataset automatically
- Train a CNN model
- Save the trained model to `models/digit_recognition_model.h5`

### 2. Launch GUI

```bash
python main.py --mode gui
```

This opens a drawing interface where you can:
- Draw digits with your mouse
- Get instant predictions
- Save your drawings

## Project Structure

```
NumVision/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training logic
│   ├── predict.py         # Prediction utilities
│   ├── draw_interface.py  # GUI drawing interface
│   ├── gpu_config.py      # GPU configuration
│   └── utils.py           # Utility functions
├── models/                # Saved models
└── tests/                 # Test images
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pillow
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Options

```bash
# Train with custom settings
python main.py --mode train --epochs 15 --batch-size 256 --learning-rate 0.001

# Train on CPU only
python main.py --mode train --no-gpu

# Disable mixed precision
python main.py --mode train --no-mixed-precision
```

## Model Architecture

- **Input**: 28×28 grayscale images
- **Conv2D**: 32 filters, 3×3 kernel
- **MaxPooling**: 2×2
- **Conv2D**: 64 filters, 3×3 kernel
- **MaxPooling**: 2×2
- **Conv2D**: 64 filters, 3×3 kernel
- **Dense**: 128 units + Dropout (0.5)
- **Output**: 10 classes (digits 0-9)

## Expected Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98.5%
- **Training Time**: 5-10 minutes on CPU, 1-2 minutes on GPU

## Dataset

Uses the MNIST dataset:
- 60,000 training images
- 10,000 test images
- Automatically downloaded on first run
- Cached at `~/.keras/datasets/mnist.npz`

## License

MIT License

