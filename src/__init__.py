"""
NumVision - Handwritten Digit Recognition System

A Python project for recognizing handwritten digits using CNNs.
"""

__version__ = '1.0.0'
__author__ = 'NumVision Team'

from .model import DigitRecognitionModel
from .predict import predict_digit, load_trained_model
from .train import train_model

__all__ = [
    'DigitRecognitionModel',
    'predict_digit',
    'load_trained_model',
    'train_model'
]

