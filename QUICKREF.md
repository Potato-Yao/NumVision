# 🚀 NumVision - Quick Reference

## ⚡ Fastest Way to Get Started

```bash
# 1. Train a model (5-10 minutes)
python quickstart.py

# 2. Draw and test digits
python src/draw_interface.py
```

## 📋 All Commands

### Training
```bash
python quickstart.py                                          # Quick training
python main.py --mode train --epochs 10 --batch-size 128     # Custom training
```

### Testing & Evaluation
```bash
python main.py --mode evaluate                                # Basic evaluation
python main.py --mode evaluate --detailed                     # Detailed results
python verify_setup.py                                        # Verify installation
python test_model.py                                          # Run unit tests
```

### Predictions
```bash
python src/draw_interface.py                                  # Draw and predict
python main.py --mode predict --image path/to/image.png      # Predict one image
python main.py --mode predict --batch "img1.png,img2.png"    # Predict multiple
python main.py --mode interactive                             # Interactive mode
```

### Demos & Examples
```bash
python demo.py                                                # Full demo
python demo.py --quick                                        # Quick demo
python demo.py --demo train                                   # Specific demo
python examples.py                                            # Usage examples
python create_test_images.py                                  # Generate test images
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `quickstart.py` | Fastest way to train a model |
| `main.py` | Main CLI with all features |
| `src/draw_interface.py` | Interactive drawing GUI |
| `examples.py` | Code usage examples |
| `demo.py` | Comprehensive demonstrations |
| `verify_setup.py` | Check installation |
| `test_model.py` | Unit tests |

## 🎯 Common Use Cases

### 1️⃣ First Time Setup
```bash
python verify_setup.py          # Verify everything works
python quickstart.py            # Train your first model
python src/draw_interface.py    # Test it interactively
```

### 2️⃣ Create & Test Images
```bash
python create_test_images.py                              # Generate test images
python main.py --mode predict --image tests/digit_5.png  # Test prediction
```

### 3️⃣ Evaluate Model Performance
```bash
python main.py --mode evaluate --detailed    # Full evaluation
# Check results/ folder for confusion matrix, reports, etc.
```

### 4️⃣ Use in Your Code
```python
from src.predict import load_trained_model, predict_digit

model = load_trained_model('models/digit_recognition_model.h5')
digit, confidence, probs = predict_digit(model, 'my_image.png')
print(f"Predicted: {digit} ({confidence*100:.2f}%)")
```

## 📊 Project Stats

- **Python Files**: 14
- **Lines of Code**: 2,500+
- **Unit Tests**: 14 (all passing ✅)
- **Model Accuracy**: ~98.5%
- **Training Time**: 5-10 minutes
- **Model Size**: ~1.5 MB

## 🔧 Customization

### Change Model Architecture
Edit `src/model.py` - modify the `build_model()` method

### Adjust Training Parameters
Edit `src/config.py` or use CLI flags:
```bash
python main.py --mode train --epochs 20 --learning-rate 0.0005
```

### Enable Data Augmentation
Edit `src/config.py`:
```python
DATA_CONFIG = {
    'augmentation': True,  # Change to True
    ...
}
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Run `python quickstart.py` first |
| Import errors | Activate venv: `.venv\Scripts\activate` |
| Dependencies missing | Run `pip install -r requirements.txt` |
| Low accuracy | Train longer, check image preprocessing |

## 📚 Documentation

- **README.md** - Full project documentation
- **GETTING_STARTED.md** - Detailed setup guide
- **PROJECT_SUMMARY.md** - Complete project overview
- **QUICKREF.md** - This file

## 🎓 Learning Path

1. ✅ Run `verify_setup.py` to check installation
2. ✅ Train model with `quickstart.py`
3. ✅ Test with `draw_interface.py`
4. ✅ Explore `examples.py`
5. ✅ Read the code in `src/` folder
6. ✅ Modify and experiment!

## 💡 Tips

- Use the drawing interface for quick testing
- Check the `results/` folder after detailed evaluation
- Run tests before making changes: `python test_model.py`
- Keep model files in `models/` folder
- Generated test images go to `tests/` folder

## 🎉 Quick Win

Want to see it work in 30 seconds?

```bash
# If you have a trained model already:
python src/draw_interface.py

# If not, train one first (takes 5-10 min):
python quickstart.py

# Then:
python src/draw_interface.py
```

---

**Need help?** Check the full documentation in README.md or GETTING_STARTED.md

**Ready to go!** 🚀

