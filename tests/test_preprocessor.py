"""
Unit tests for preprocessor module.
Tests image preprocessing functionality.
"""

import cv2
import numpy as np
from cv_pipeline.preprocessor import preprocess, preprocess_advanced


def test_preprocess_returns_tuple():
    """Test that preprocess returns a tuple of 3 images."""
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = preprocess(dummy_image)
    
    assert isinstance(result, tuple), "preprocess should return a tuple"
    assert len(result) == 3, "preprocess should return 3 images"
    assert isinstance(result[0], np.ndarray), "First element should be numpy array"
    assert isinstance(result[1], np.ndarray), "Second element should be numpy array"
    assert isinstance(result[2], np.ndarray), "Third element should be numpy array"
    print("✓ test_preprocess_returns_tuple passed")


def test_preprocess_output_shapes():
    """Test that output images have correct shapes."""
    dummy_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    preprocessed, gray, thresh = preprocess(dummy_image)
    
    assert preprocessed.shape[:2] == (200, 300), "Output should have same height/width"
    assert gray.shape == (200, 300), "Grayscale should be 2D"
    assert thresh.shape == (200, 300), "Threshold should be 2D"
    print("✓ test_preprocess_output_shapes passed")


def test_preprocess_output_values():
    """Test that preprocessed output is binary (0-255)."""
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    preprocessed, _, _ = preprocess(dummy_image)
    
    assert preprocessed.min() >= 0, "Minimum value should be >= 0"
    assert preprocessed.max() <= 255, "Maximum value should be <= 255"
    assert preprocessed.dtype == np.uint8, "Output should be uint8"
    print("✓ test_preprocess_output_values passed")


def test_preprocess_advanced():
    """Test advanced preprocessing with custom parameters."""
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = preprocess_advanced(dummy_image, block_size=15, constant=3)
    
    assert isinstance(result, np.ndarray), "Should return numpy array"
    assert result.shape[:2] == (100, 100), "Shape should be preserved"
    print("✓ test_preprocess_advanced passed")


def test_preprocess_with_real_like_image():
    """Test preprocessing with realistic image patterns."""
    # Create an image with some patterns (simulating receipt)
    dummy_image = np.ones((200, 300, 3), dtype=np.uint8) * 150
    cv2.rectangle(dummy_image, (50, 50), (250, 150), (200, 200, 200), -1)
    cv2.putText(dummy_image, "EGGS", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    preprocessed, gray, thresh = preprocess(dummy_image)
    
    assert preprocessed.shape[:2] == (200, 300), "Shape should match input"
    # Check that preprocessing actually changed something
    assert not np.array_equal(preprocessed, dummy_image), "Output should be different from input"
    print("✓ test_preprocess_with_real_like_image passed")


if __name__ == "__main__":
    test_preprocess_returns_tuple()
    test_preprocess_output_shapes()
    test_preprocess_output_values()
    test_preprocess_advanced()
    test_preprocess_with_real_like_image()
    print("\n✓ All preprocessor tests passed!")
