Here is a concise 6‑minute presentation script you can adapt. I’ll assume `NumVision` is a TensorFlow\-based computer vision project (number/handwritten digit recognition, etc.). Adjust wording to match your exact task.

---

## 1\. Project overview \(\~1 min\)

Hi everyone, today we’ll present our project `NumVision`.  
This is a deep learning project built with Python and TensorFlow, focused on digit recognition from images.

The main goals are:

- Build a CNN model that can learn from image data.
- Organize the code so training, evaluation, and configuration are cleanly separated.
- Provide utilities to run efficiently on GPU and make it easy to adjust and experiment with models.

---

## 2\. Project structure \(\~2 min\)

The project is organized in a fairly standard way:

- `src/` – all Python source code
    - `gpu_config.py` – utilities to detect and configure GPUs
    - `data_*.py` or `datasets/` – data loading and preprocessing
    - `models/` – model definitions (e.g. CNN, ResNet\-like models)
    - `train.py` – main training script
    - `eval.py` or `inference.py` – evaluation / prediction script
- `requirements.txt` – Python and TensorFlow dependencies
- `README.md` – short project description and how to run it

**Focus on `gpu_config.py`**  
This file is responsible for making training efficient and stable on machines with NVIDIA GPUs:

- `check_gpu_availability()`
    - Prints TensorFlow version.
    - Lists physical GPU devices.
    - Shows CUDA/cuDNN status and device name.
    - Returns a boolean to tell the rest of the code if GPU is available.

- `configure_gpu_memory_growth()`
    - By default TensorFlow can grab all GPU memory.
    - This function enables *memory growth* so memory is allocated gradually, which prevents conflicts and out\-of\-memory errors when sharing a GPU.

- `set_mixed_precision(enable=True)`
    - Turns on mixed precision \(float16 for compute, float32 for variables\).
    - On modern GPUs, this can give 2–3x speedup and lower memory use.

- `configure_for_training(use_mixed_precision=True, memory_growth=True)`
    - High\-level function called before training.
    - Combines GPU check, memory growth, and mixed precision into one step.
    - Returns whether a GPU is available, so the training script can decide how to proceed.

- Utility helpers:
    - `limit_gpu_memory(memory_limit_mb)` – caps GPU memory usage to a fixed size.
    - `get_gpu_memory_info()` – returns a small dict with GPU availability and names.
    - `set_gpu_device(device_id=0)` – selects a specific GPU on multi\-GPU systems.

We use class to organize the model logic:

DigitRecognitionModel is a wrapper class around a TensorFlow/Keras CNN specifically designed for handwritten digit recognition (e.g. MNIST).
It encapsulates the full workflow: building, compiling, training, evaluating, predicting, saving, and loading the model.
build_model\(\) creates a 3-block convolutional network followed by dense layers and dropout, outputting 10 classes (digits 0–9).
compile_model\(\) configures the optimizer (Adam), loss (sparse_categorical_crossentropy), and accuracy metric.
train\(\) adds training utilities like early stopping, model checkpoints, and learning rate scheduling.
evaluate\(\) reports test loss and accuracy.
predict\(\) returns predicted digit labels and their probabilities.
save_model\(\) / load_model\(\) handle persistence.
get_training_history\(\) exposes the metrics history for plotting or analysis.

There are four important functions we want to highlight:

check_gpu_availability()
Checks if TensorFlow can see any GPU, prints TF/GPU info, and returns a boolean flag for GPU availability.

configure_for_training(use_mixed_precision=True, memory_growth=True)
High-level setup used at the start of training: checks GPU, enables memory growth, and optionally turns on mixed precision.

build_cnn_model(input_shape, num_classes, base_filters=32, dropout_rate=0.3)
Builds the main CNN architecture used for digit recognition, with adjustable capacity via base_filters and dropout_rate.

load_dataset(batch_size, image_size)
Loads and preprocesses the image data, returning tf.data training and validation datasets with normalization and augmentations.

---

## 3\. How the project works end\-to\-end \(\~2 min\)

**1\. Entry point: training script**

In `train.py`, the rough flow is:

1. Import and run GPU configuration:
   ```python
   # python
   from src.gpu_config import configure_for_training

   gpu_ok = configure_for_training(
       use_mixed_precision=True,
       memory_growth=True,
   )
   ```
    - This prepares the hardware and prints device info.
    - If no GPU is found, the code falls back to CPU training.

2. Load and preprocess data:
   ```python
   # python
   train_ds, val_ds = load_dataset(
       batch_size=64,
       image_size=(28, 28),
   )
   ```
    - Uses `tf.data` pipelines for performance.
    - Includes normalization, reshaping, and possibly augmentation.

3. Build the model:
   ```python
   # python
   model = build_cnn_model(
       input_shape=(28, 28, 1),
       num_classes=10,
   )
   ```
    - Typically a stack of Conv2D, pooling, and dense layers.
    - If mixed precision is enabled, the model automatically uses `mixed_float16` where appropriate.

4. Compile and train:
   ```python
   # python
   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss="sparse_categorical_crossentropy",
       metrics=["accuracy"],
   )

   history = model.fit(
       train_ds,
       validation_data=val_ds,
       epochs=10,
   )
   ```
    - Training automatically uses GPU if available, otherwise CPU.
    - Mixed precision and memory growth are already active.

5. Save the model or evaluate:
   ```python
   # python
   model.save("artifacts/numvision_model")
   ```

---

## 4\. Examples: how we adjust and experiment with the model \(\~1 min\)

There are several simple levers we can use to adjust the model and training without touching low\-level details.

### 4\.1 Adjust GPU behavior

In `train.py`, we can quickly switch configurations:

```python
# python
from src.gpu_config import configure_for_training, limit_gpu_memory

gpu_ok = configure_for_training(
    use_mixed_precision=False,  # turn off mixed precision if there are stability issues
    memory_growth=True,
)

# If sharing GPU with others, cap memory to 4096 MB
limit_gpu_memory(4096)
```

- This is useful on shared servers or when debugging on older GPUs.

### 4\.2 Change model capacity

In a `models/cnn.py` file, we might have:

```python
# python
def build_cnn_model(input_shape, num_classes, base_filters=32, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(base_filters, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(base_filters * 2, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(base_filters * 4, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="numvision_cnn")
```

To experiment:

- Increase `base_filters` to 64 for a larger, more expressive model.
- Increase `dropout_rate` if overfitting.
- Reduce them if training is too slow.

### 4\.3 Change training hyperparameters

In `train.py`:

```python
# python
BATCH_SIZE = 128          # was 64
EPOCHS = 20               # was 10
LEARNING_RATE = 5e-4      # adjust learning rate

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
```

- Larger batch size uses GPU memory more but can speed up training.
- More epochs let the model converge further.
- Learning rate controls how fast or stable training is.

---

## 5\. Summary \(\~30 sec\)

- The project is organized modularly: separate files for GPU configuration, data, models, and training.
- `gpu_config.py` abstracts away complex GPU setup, enabling:
    - Device detection
    - Memory growth
    - Mixed precision
    - Memory limiting and device selection
- Training follows a clear pipeline: configure hardware → load data → build model → train → save/evaluate.
- Model adjustments are straightforward: change GPU settings, model size \(filters, layers\), and training hyperparameters.

This structure makes `NumVision` easier to maintain, extend, and experiment with, especially when moving between CPU and GPU environments.
