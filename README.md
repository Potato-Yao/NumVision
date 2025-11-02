# NumVision - Handwritten Digit Recognition

A Python project for recognizing handwritten digits (0-9) using a custom-trained Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Features

- **Custom CNN Model**: Trained from scratch on the MNIST dataset
- **High Accuracy**: Achieves >98% accuracy on test data
- **Real-time Prediction**: Test the model with your own digit images
- **Visualization**: View training history and sample predictions
- **Easy to Use**: Simple command-line interface

## Project Structure

```
NumVision/
├── src/
│   ├── model.py          # CNN model architecture
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction utilities
│   └── utils.py          # Helper functions
├── models/               # Saved trained models
├── data/                 # Dataset storage
├── tests/                # Test images
├── requirements.txt      # Dependencies
├── main.py              # Main entry point
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd NumVision
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python main.py --mode train --epochs 10 --batch-size 128
```

### Evaluate the Model

```bash
python main.py --mode evaluate
```

### Make Predictions

```bash
python main.py --mode predict --image path/to/your/image.png
```

### Interactive Mode

```bash
python main.py --mode interactive
```

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers (32 and 64 filters)
- Max Pooling layers
- Dropout for regularization
- Dense layers with 128 units
- Softmax output layer (10 classes)

## Performance

- Training Accuracy: ~99%
- Test Accuracy: ~98.5%
- Training Time: ~5 minutes on CPU

## Dataset

The project uses the MNIST dataset:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes (digits 0-9)

## License

MIT License

## Author

Created with NumVision
numpy>=1.24.0
tensorflow>=2.13.0
matplotlib>=3.7.0
pillow>=10.0.0
scikit-learn>=1.3.0

