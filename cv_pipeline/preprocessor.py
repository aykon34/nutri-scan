"""
Image Preprocessing Module - Technique 1 of 3

Handles image preprocessing including:
- Grayscale conversion
- Adaptive thresholding (handles uneven lighting on crumpled receipts)
- Morphological operations (dilation/erosion) to clean up noise

This technique is crucial for improving OCR accuracy on receipt images.
"""

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> tuple:
    """
    Preprocess receipt image for improved OCR accuracy.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
    
    Returns:
        tuple: (preprocessed_image, grayscale_image, threshold_image)
               - preprocessed_image: Final cleaned image ready for OCR
               - grayscale_image: For visualization of step 1
               - threshold_image: For visualization of step 2
    """
    
    # Step 1: Convert to Grayscale
    # Handles RGB to single channel conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian Blur
    # Reduces noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Apply Adaptive Thresholding
    # More robust than fixed threshold for receipts with uneven lighting
    # Calculates threshold for each pixel based on 11x11 neighborhood
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size (must be odd)
        2    # Constant subtracted from mean
    )
    
    # Step 4: Morphological Operations
    # Use a small kernel to avoid merging thin receipt text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Apply closing (fills tiny gaps inside characters without merging them)
    # Using only 1 iteration to preserve character strokes
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned, gray, thresh


def preprocess_advanced(image: np.ndarray, 
                       block_size: int = 11,
                       constant: int = 2,
                       dilation_iter: int = 2,
                       erosion_iter: int = 1) -> np.ndarray:
    """
    Advanced preprocessing with configurable parameters.
    
    Args:
        image: Input image
        block_size: Size of the neighborhood area (must be odd)
        constant: Constant subtracted from mean
        dilation_iter: Number of dilation iterations
        erosion_iter: Number of erosion iterations
    
    Returns:
        Preprocessed image
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned