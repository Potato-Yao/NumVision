"""
Configuration file for NumVision project.
"""

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 10,
    'architecture': 'cnn',
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 128,
    'learning_rate': 0.001,
    'validation_split': 0.1,
    'early_stopping_patience': 3,
    'reduce_lr_patience': 2,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
}

# Data Configuration
DATA_CONFIG = {
    'dataset': 'mnist',
    'normalize': True,
    'augmentation': False,  # Set to True for data augmentation
    'augmentation_params': {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.1,
    }
}

# Paths Configuration
PATHS = {
    'models_dir': 'models',
    'data_dir': 'data',
    'tests_dir': 'tests',
    'results_dir': 'results',
    'model_save_path': 'models/digit_recognition_model.h5',
    'best_model_path': 'models/best_model.h5',
    'training_history_plot': 'models/training_history.png',
    'predictions_plot': 'models/predictions_sample.png',
}

# Visualization Configuration
VIZ_CONFIG = {
    'plot_dpi': 300,
    'plot_style': 'seaborn',
    'figure_size': (12, 6),
    'cmap': 'gray',
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'confidence_threshold': 0.5,
    'top_k': 3,
    'show_plot': True,
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'min_accuracy': 0.95,
    'target_accuracy': 0.98,
    'max_training_time_minutes': 15,
}

