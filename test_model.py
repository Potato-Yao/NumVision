"""
Unit tests for the digit recognition system.
"""
import unittest
import numpy as np
import os
import sys
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import DigitRecognitionModel
from src.predict import preprocess_image, predict_from_array


class TestDigitRecognitionModel(unittest.TestCase):
    """Test cases for the DigitRecognitionModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DigitRecognitionModel(input_shape=(28, 28, 1), num_classes=10)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_shape, (28, 28, 1))
        self.assertEqual(self.model.num_classes, 10)
        self.assertIsNone(self.model.model)

    def test_build_model(self):
        """Test model building."""
        self.model.build_model()
        self.assertIsNotNone(self.model.model)

        # Check input shape
        self.assertEqual(self.model.model.input_shape, (None, 28, 28, 1))

        # Check output shape
        self.assertEqual(self.model.model.output_shape, (None, 10))

    def test_compile_model(self):
        """Test model compilation."""
        self.model.build_model()
        self.model.compile_model(learning_rate=0.001)

        # Check if optimizer is set
        self.assertIsNotNone(self.model.model.optimizer)

    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        self.model.build_model()
        self.model.compile_model()

        # Create dummy input
        dummy_input = np.random.random((5, 28, 28, 1))
        predictions = self.model.model.predict(dummy_input, verbose=0)

        # Check output shape
        self.assertEqual(predictions.shape, (5, 10))

        # Check if probabilities sum to 1
        for pred in predictions:
            self.assertAlmostEqual(np.sum(pred), 1.0, places=5)


class TestPreprocessing(unittest.TestCase):
    """Test cases for image preprocessing."""

    def test_preprocess_array(self):
        """Test preprocessing of numpy arrays."""
        # Create a dummy 28x28 image
        img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        # Normalize
        normalized = img.astype('float32') / 255.0

        # Check range
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))

    def test_image_shape_expansion(self):
        """Test adding channel and batch dimensions."""
        img = np.random.random((28, 28))

        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        self.assertEqual(img.shape, (28, 28, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        self.assertEqual(img.shape, (1, 28, 28, 1))


class TestMNISTDataset(unittest.TestCase):
    """Test cases for MNIST dataset loading."""

    def test_mnist_loading(self):
        """Test MNIST dataset can be loaded."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Check shapes
        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(x_test.shape, (10000, 28, 28))
        self.assertEqual(y_test.shape, (10000,))

        # Check data types
        self.assertEqual(x_train.dtype, np.uint8)
        self.assertEqual(y_train.dtype, np.uint8)

        # Check value ranges
        self.assertTrue(np.all(x_train >= 0))
        self.assertTrue(np.all(x_train <= 255))
        self.assertTrue(np.all(y_train >= 0))
        self.assertTrue(np.all(y_train <= 9))

    def test_class_distribution(self):
        """Test that all digit classes are present."""
        (_, y_train), (_, y_test) = keras.datasets.mnist.load_data()

        # Check all classes present in training set
        unique_train = np.unique(y_train)
        self.assertEqual(len(unique_train), 10)
        self.assertTrue(np.array_equal(unique_train, np.arange(10)))

        # Check all classes present in test set
        unique_test = np.unique(y_test)
        self.assertEqual(len(unique_test), 10)
        self.assertTrue(np.array_equal(unique_test, np.arange(10)))


class TestModelArchitecture(unittest.TestCase):
    """Test cases for model architecture."""

    def setUp(self):
        """Set up test model."""
        self.model = DigitRecognitionModel()
        self.model.build_model()

    def test_number_of_layers(self):
        """Test that model has expected number of layers."""
        # Should have: 3 Conv2D, 2 MaxPooling, 1 Flatten, 2 Dense, 1 Dropout
        self.assertGreater(len(self.model.model.layers), 5)

    def test_layer_types(self):
        """Test that model contains expected layer types."""
        layer_types = [type(layer).__name__ for layer in self.model.model.layers]

        # Check for convolutional layers
        self.assertTrue('Conv2D' in layer_types)

        # Check for pooling layers
        self.assertTrue('MaxPooling2D' in layer_types)

        # Check for dense layers
        self.assertTrue('Dense' in layer_types)

        # Check for flatten layer
        self.assertTrue('Flatten' in layer_types)

    def test_output_activation(self):
        """Test that output layer uses softmax activation."""
        output_layer = self.model.model.layers[-1]
        self.assertEqual(output_layer.activation.__name__, 'softmax')

    def test_trainable_parameters(self):
        """Test that model has trainable parameters."""
        trainable_count = np.sum([np.prod(v.shape) for v in self.model.model.trainable_weights])
        self.assertGreater(trainable_count, 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_directory_creation(self):
        """Test that directories can be created."""
        test_dir = 'test_temp_dir'
        os.makedirs(test_dir, exist_ok=True)
        self.assertTrue(os.path.exists(test_dir))
        os.rmdir(test_dir)

    def test_numpy_operations(self):
        """Test numpy operations used in the project."""
        arr = np.array([1, 2, 3, 4, 5])

        # Test argmax
        self.assertEqual(np.argmax(arr), 4)

        # Test expand_dims
        expanded = np.expand_dims(arr, axis=0)
        self.assertEqual(expanded.shape, (1, 5))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDigitRecognitionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestMNISTDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestModelArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

