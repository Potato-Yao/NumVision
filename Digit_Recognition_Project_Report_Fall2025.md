# NumVision: Handwritten Digit Recognition with CNN and Interactive Drawing Interface

## Title Page
Project Title: NumVision – Real-Time Handwritten Digit Recognition and Interaction Platform  
Course: Programming with Python (Fall 2025)  
Group Members: <姚雨森 – 1120242312>, <曾梓木 – 1120242250>
Date: November 9, 2025  

---
## 1. Introduction & Motivation
Handwritten digit recognition is a foundational problem in computer vision and machine learning, powering applications such as postal code sorting, bank check processing, form digitization, and educational tools. While benchmark performance on the MNIST dataset is now saturated, integrating high-accuracy models into user-friendly, real-time systems remains pedagogically valuable and practical.

This project builds a clean, reproducible convolutional neural network (CNN) pipeline for MNIST digit classification and extends it with an interactive Tkinter-based drawing interface that allows users to draw digits and obtain instantaneous predictions with confidence scores and top-3 alternatives. We also incorporate GPU optimizations and training stability enhancements for reproducibility under constrained hardware environments.

---
## 2. Problem Definition
Goal: Develop a robust, efficient, and user-interactive system that correctly classifies handwritten digits (0–9) from user-provided drawings or image files with high accuracy (≥99% test accuracy on MNIST), while ensuring:
- Reproducible training pipeline.
- Real-time inference for single-digit drawings.
- Model performance transparency (classification report, confusion matrix, misclassification analysis).

Key Questions:
1. Can a compact CNN architecture achieve ≥99% test accuracy using standard training protocols without heavy augmentation?
2. What training interventions (early stopping, learning rate reduction, mixed precision) improve stability and efficiency?
3. How does the model behave on edge-case digits or ambiguous user drawings?

---
## 3. Proposed Method
We implement a custom CNN tailored to MNIST with a balance between depth and computational simplicity. The architecture emphasizes progressively increasing channel depth and spatial reduction via pooling, followed by dense layers for classification.

### Model Architecture
Layers (in order):
1. Conv2D (32 filters, 3×3, ReLU, same padding)
2. MaxPooling2D (2×2)
3. Conv2D (64 filters, 3×3, ReLU, same padding)
4. MaxPooling2D (2×2)
5. Conv2D (64 filters, 3×3, ReLU, same padding)
6. Flatten
7. Dense (128 units, ReLU)
8. Dropout (rate = 0.5)
9. Dense (10 units, Softmax)

Parameter Count (calculated):
- Conv1: (3×3×1×32)+32 = 320
- Conv2: (3×3×32×64)+64 = 18,496
- Conv3: (3×3×64×64)+64 = 36,928
- Flatten size: 7×7×64 = 3,136
- Dense1: (3,136×128)+128 = 401,536
- Output: (128×10)+10 = 1,290  
Total Trainable Parameters ≈ 458,570

### Training Enhancements
- EarlyStopping (patience=3, monitor=val_loss, restore best weights).
- ModelCheckpoint (monitor=val_accuracy, save best model to `models/best_model.h5`).
- ReduceLROnPlateau (monitor=val_loss, factor=0.5, patience=2, min_lr=1e-7).
- Optional Mixed Precision (float16/float32) if compatible GPU.
- GPU memory growth enabled to avoid OOM on shared systems.

### Data Preprocessing
- Source: MNIST via `keras.datasets.mnist`.
- Normalization: Pixel intensities scaled to [0, 1].
- Reshape: (28,28) → (28,28,1).
- Train/Val Split: 90% / 10% stratified.

### Novelty Elements
- Integrated real-time digit drawing and prediction interface (`src/draw_interface.py`).
- Automated GPU configuration and mixed precision management (`src/gpu_config.py`).
- Confidence and top-3 probability reporting for interpretability.

---
## 4. Algorithm Description
Let X ∈ R^{N×28×28×1} be input images and Y ∈ {0,…,9}^N the labels.

Forward Pass:
1. Convolutional feature extraction through stacked Conv/ReLU blocks.
2. Spatial reduction via MaxPooling.
3. High-level feature vector via flattening.
4. Non-linear transformation and regularization (Dense + Dropout).
5. Final classification via softmax probabilities.

Loss: Sparse categorical cross-entropy: L = −log p(y|x).  
Optimizer: Adam (learning_rate = 0.001 initial, dynamically reduced).  
Metrics: Accuracy (train/val/test); per-class precision, recall, F1 via scikit-learn post-training.

Regularization: Dropout (0.5), early stopping to mitigate overfitting, LR scheduling for smoother convergence.

---
## 5. Experiments
### 5.1 Experimental Setup
- Hardware: CPU-only or GPU (if available; mixed precision enabled automatically).
- Frameworks: TensorFlow/Keras, NumPy, scikit-learn, Pillow, Tkinter.
- Epochs: Up to 10 (early stopping may reduce effective epochs).
- Batch Size: 128 (auto-scaled to 256 if GPU detected).
- Validation Split: 10% stratified.

### 5.2 Hyperparameters
- Learning Rate: 0.001 (adaptive reduction via plateau scheduler).
- Dropout: 0.5 in dense layer.
- Patience (EarlyStopping): 3.
- LR Reduction Factor: 0.5.

### 5.3 Training Dynamics
Training history (see `models/training_history.png`) shows rapid convergence within first few epochs. Validation loss stabilized before epoch limit due to early stopping.

### 5.4 Results
Final Test Set Accuracy: ≈ 99% (See classification report below).

Classification Report (excerpt, full in `results/classification_report.txt`):
- Macro Avg Precision: 0.99
- Macro Avg Recall: 0.99
- Macro Avg F1: 0.99
- Weighted Avg Accuracy: 0.99 (10,000 test samples)

Confusion Matrix: Refer to `results/confusion_matrix.png`. Sparse misclassifications occur primarily among visually similar digits (e.g., 5 vs 3, 9 vs 4 when stroke thickness or closure is ambiguous).

Misclassifications: `results/misclassifications.png` highlights ambiguous samples—often low stroke contrast or partial digit formation.

### 5.5 Robustness Considerations
- Mixed precision improved throughput on supporting GPUs without accuracy degradation.
- Early stopping prevented overfitting (no significant divergence between training and validation accuracy).
- No explicit data augmentation used; adding augmentation could further generalize to non-MNIST styles (e.g., user-drawn digits with varied stroke thickness).

### 5.6 Ablation Opportunities (Proposed, Not Executed)
Potential studies to include if time permits:
- Remove Dropout: Observe overfitting speed.
- Replace Adam with SGD+Momentum: Compare convergence stability.
- Add BatchNormalization after conv layers.
- Add data augmentation (random shifts, elastic distortions).

### 5.7 Reproducibility Steps
1. Create environment (Python ≥3.10 recommended).  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run training: `python -m src.train`  
4. Launch drawing UI (after training): `python -c "from tensorflow import keras; from src.draw_interface import launch_drawing_app; m=keras.models.load_model('models/digit_recognition_model.h5'); launch_drawing_app(m)"`  
5. Evaluate predictions with test images via `src.predict.py` utilities.

Environment Determinism Suggestions:
- Set seeds (future enhancement): `tf.random.set_seed(42); np.random.seed(42)`.
- Pin exact versions in `requirements.txt` (already minimally constrained).

---
## 6. Discussion & Conclusion
### Key Findings
- A moderately sized CNN can achieve ≥99% accuracy on MNIST without augmentation or architectural complexity.
- Training stability is enhanced by early stopping and adaptive LR scheduling; both are low-effort high-impact additions.
- Mixed precision (where supported) improves training efficiency without harming convergence—important for scalability.
- Interactive interface increases accessibility, demonstrating model deployment and human-model feedback loops.

### Strengths
- High accuracy with simple architecture.
- Reproducible and well-commented codebase.
- User-centric demonstration via drawing interface.
- Modular design (separate GPU config, model class, prediction utilities).

### Limitations
- No custom dataset integration beyond canonical MNIST.
- Lack of augmentation may limit generalization to unconstrained handwritten styles.
- Single-model approach; no ensemble robustness.

### Future Directions
- Add data augmentation (affine transforms, elastic distortions).
- Deploy as a lightweight web app (e.g., FastAPI + ONNX runtime).
- Quantize model for edge devices (TensorFlow Lite conversion).
- Explore adversarial robustness (FGSM evaluation).
- Extend interface for multi-digit sequence parsing (segmentation + sequence modeling).
- Add uncertainty estimation (Monte Carlo Dropout or temperature scaling).

Conclusion: The NumVision project successfully delivers a performant and user-interactive handwritten digit recognition system. It fulfills the course requirements—clear problem definition, use of machine learning libraries, more than 100 lines of well-documented code, and practical novelty via real-time interaction and GPU optimization.

---
## 7. References
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
2. Deng, L. (2012). The MNIST database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine.
3. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
4. TensorFlow Documentation: https://www.tensorflow.org/  
5. Keras API Reference: https://keras.io/  
6. Scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html  
7. Mixed Precision Training Guide (TensorFlow): https://www.tensorflow.org/guide/mixed_precision  

---
## Appendix A: File Overview
- `src/model.py`: CNN definition and training/evaluation methods.
- `src/train.py`: End-to-end training pipeline with callbacks.
- `src/predict.py`: Single and batch prediction utilities.
- `src/draw_interface.py`: Tkinter interactive digit drawing and prediction UI.
- `src/gpu_config.py`: GPU detection and mixed precision configuration.
- `results/`: Generated artifacts (classification report, confusion matrix, misclassifications, training history).
- `models/`: Saved best and final trained models.

## Appendix B: Potential Reproducibility Enhancements
- Add explicit seeding and deterministic ops flags.
- Provide Dockerfile for containerized execution.
- Log training metrics to JSON/CSV for external analysis.

---
Prepared by: <Group Members>  
End of Report.

