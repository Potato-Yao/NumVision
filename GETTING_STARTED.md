# NumVision Project - Getting Started Guide

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment (if not already activated)
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Train Your First Model

The easiest way to get started is using the quickstart script:

```bash
python quickstart.py
```

This will:
- Download the MNIST dataset automatically
- Train a CNN model for ~5-10 minutes
- Save the model to `models/digit_recognition_model.h5`
- Generate training visualizations

### 3. Test the Model

After training, you can test the model in several ways:

#### A. Evaluate on Test Set
```bash
python main.py --mode evaluate --detailed
```

#### B. Create Test Images
```bash
python create_test_images.py
```

#### C. Predict on an Image
```bash
python main.py --mode predict --image tests/digit_5_simple.png
```

#### D. Interactive Drawing Interface
```bash
python src/draw_interface.py
```
Draw digits with your mouse and get real-time predictions!

#### E. Interactive Mode
```bash
python main.py --mode interactive
```

## Project Structure

```
NumVision/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── model.py              # CNN model implementation
│   ├── train.py              # Training functions
│   ├── predict.py            # Prediction utilities
│   ├── utils.py              # Helper functions
│   ├── config.py             # Configuration settings
│   └── draw_interface.py     # Drawing GUI
├── models/                   # Saved models (created automatically)
├── data/                     # Dataset cache (created automatically)
├── tests/                    # Test images (created automatically)
├── results/                  # Evaluation results (created automatically)
├── main.py                   # Main entry point
├── quickstart.py             # Quick training script
├── demo.py                   # Comprehensive demo
├── test_model.py             # Unit tests
├── create_test_images.py     # Generate test images
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── GETTING_STARTED.md        # This file
```

## Usage Examples

### Train with Custom Parameters
```bash
python main.py --mode train --epochs 15 --batch-size 256 --learning-rate 0.0005
```

### Batch Prediction
```bash
python main.py --mode predict --batch "img1.png,img2.png,img3.png"
```

### Run Full Demo
```bash
python demo.py
```

### Run Quick Demo (No Training)
```bash
python demo.py --quick
```

### Run Specific Demo
```bash
python demo.py --demo train
python demo.py --demo eval
python demo.py --demo predict
python demo.py --demo viz
python demo.py --demo draw
```

## Model Architecture

The CNN model consists of:

1. **Convolutional Layers**:
   - Conv2D: 32 filters, 3x3 kernel
   - MaxPooling2D: 2x2
   - Conv2D: 64 filters, 3x3 kernel
   - MaxPooling2D: 2x2
   - Conv2D: 64 filters, 3x3 kernel

2. **Dense Layers**:
   - Flatten
   - Dense: 128 units + ReLU
   - Dropout: 0.5
   - Dense: 10 units + Softmax

**Total Parameters**: ~93,000 trainable parameters

## Expected Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98.5%
- **Training Time**: 5-10 minutes on CPU
- **Inference Time**: <10ms per image

## Tips and Tricks

### 1. Prepare Your Own Images
For best results with your own digit images:
- Use a white/light background with black/dark digits
- Center the digit in the image
- Use clear, legible handwriting
- Avoid too much extra space around the digit

### 2. Improve Model Performance
To potentially improve accuracy:
- Increase epochs (15-20)
- Enable data augmentation in `src/config.py`
- Experiment with different learning rates
- Try different batch sizes

### 3. Save and Load Models
```python
from src.model import DigitRecognitionModel

# Save
model.save_model('my_model.h5')

# Load
model.load_model('my_model.h5')
```

### 4. Use the Model in Your Code
```python
from src.predict import load_trained_model, predict_digit

# Load model
model = load_trained_model('models/digit_recognition_model.h5')

# Predict
digit, confidence, probs = predict_digit(model, 'my_digit.png')
print(f"Predicted: {digit} with {confidence*100:.2f}% confidence")
```

## Troubleshooting

### Issue: Model not found
**Solution**: Train a model first using `python quickstart.py`

### Issue: Import errors
**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Low accuracy
**Solution**: 
- Train for more epochs
- Check that images are preprocessed correctly
- Ensure digits are centered and clear

### Issue: Slow training
**Solution**:
- Reduce batch size if running out of memory
- Consider using a GPU if available
- Reduce number of epochs for testing

## Additional Resources

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras API: https://keras.io/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

## Testing

Run the unit tests to verify everything is working:

```bash
python test_model.py
```

All 14 tests should pass.

## Next Steps

1. ✅ Train your first model: `python quickstart.py`
2. ✅ Test with the drawing interface: `python src/draw_interface.py`
3. ✅ Create your own test images: `python create_test_images.py`
4. ✅ Evaluate the model: `python main.py --mode evaluate --detailed`
5. ✅ Explore the code and modify the architecture in `src/model.py`

## Support

For issues or questions:
1. Check this guide
2. Review the README.md
3. Check the code comments
4. Run unit tests to verify setup

Happy digit recognition! 🎉

