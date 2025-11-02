"""
Utility functions for the digit recognition project.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def create_project_directories():
    """Create necessary project directories."""
    directories = ['models', 'data', 'tests', 'results']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' ready")


def plot_sample_images(x_data, y_data, num_samples=25, title='Sample Images'):
    """
    Display a grid of sample images from the dataset.

    Args:
        x_data: Image data
        y_data: Labels
        num_samples: Number of samples to display
        title: Plot title
    """
    num_samples = min(num_samples, len(x_data))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axes = axes.ravel() if num_samples > 1 else [axes]

    for i in range(num_samples):
        idx = np.random.randint(0, len(x_data))
        axes[i].imshow(x_data[idx].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {y_data[idx]}', fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, x_test, y_test, save_path='results/confusion_matrix.png'):
    """
    Generate and plot confusion matrix.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
        save_path: Path to save the plot
    """
    # Make predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

    return cm


def generate_classification_report(model, x_test, y_test, save_path='results/classification_report.txt'):
    """
    Generate detailed classification report.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
        save_path: Path to save the report
    """
    # Make predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Generate report
    report = classification_report(y_test, predicted_classes,
                                   target_names=[f'Digit {i}' for i in range(10)])

    print("\nClassification Report:")
    print(report)

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)

    print(f"Classification report saved to {save_path}")

    return report


def analyze_misclassifications(model, x_test, y_test, num_examples=10,
                               save_path='results/misclassifications.png'):
    """
    Visualize misclassified examples.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
        num_examples: Number of misclassified examples to show
        save_path: Path to save the plot
    """
    # Make predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Find misclassified indices
    misclassified_idx = np.where(predicted_classes != y_test)[0]

    if len(misclassified_idx) == 0:
        print("No misclassifications found!")
        return

    print(f"Total misclassifications: {len(misclassified_idx)} out of {len(y_test)}")

    # Select random misclassified examples
    num_examples = min(num_examples, len(misclassified_idx))
    selected_idx = np.random.choice(misclassified_idx, num_examples, replace=False)

    # Plot
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.ravel()

    for i, idx in enumerate(selected_idx):
        if i >= rows * cols:
            break

        axes[i].imshow(x_test[idx].squeeze(), cmap='gray')
        pred_label = predicted_classes[idx]
        true_label = y_test[idx]
        confidence = predictions[idx][pred_label] * 100

        axes[i].set_title(
            f'True: {true_label}, Pred: {pred_label}\n'
            f'Conf: {confidence:.1f}%',
            color='red',
            fontsize=10
        )
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(selected_idx), rows * cols):
        axes[i].axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Misclassifications plot saved to {save_path}")
    plt.close()


def get_model_info(model):
    """
    Print detailed model information.

    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)

    # Count parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")

    print(f"\nInput Shape: {model.input_shape}")
    print(f"Output Shape: {model.output_shape}")

    print(f"\nNumber of Layers: {len(model.layers)}")
    print("\nLayer Details:")
    for i, layer in enumerate(model.layers):
        print(f"  {i+1}. {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'output_shape'):
            print(f"     Output Shape: {layer.output_shape}")

    print("="*60 + "\n")


def save_test_results(model, x_test, y_test, save_dir='results'):
    """
    Generate and save all test results.

    Args:
        model: Trained model
        x_test: Test images
        y_test: True labels
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating comprehensive test results...")

    # Confusion matrix
    plot_confusion_matrix(model, x_test, y_test,
                         save_path=f'{save_dir}/confusion_matrix.png')

    # Classification report
    generate_classification_report(model, x_test, y_test,
                                   save_path=f'{save_dir}/classification_report.txt')

    # Misclassifications
    analyze_misclassifications(model, x_test, y_test,
                               save_path=f'{save_dir}/misclassifications.png')

    print(f"\nAll results saved to '{save_dir}/' directory")


def data_augmentation_preview(x_data, y_data, num_samples=5):
    """
    Preview original and augmented images.

    Args:
        x_data: Image data
        y_data: Labels
        num_samples: Number of samples to show
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    fig, axes = plt.subplots(num_samples, 5, figsize=(12, num_samples*2))

    for i in range(num_samples):
        idx = np.random.randint(0, len(x_data))
        img = x_data[idx:idx+1]
        label = y_data[idx]

        # Original
        axes[i, 0].imshow(img.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original: {label}')
        axes[i, 0].axis('off')

        # Augmented versions
        aug_iter = datagen.flow(img, batch_size=1)
        for j in range(1, 5):
            aug_img = next(aug_iter)
            axes[i, j].imshow(aug_img.squeeze(), cmap='gray')
            axes[i, j].set_title(f'Augmented {j}')
            axes[i, j].axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    create_project_directories()

