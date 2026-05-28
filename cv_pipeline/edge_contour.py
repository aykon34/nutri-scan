"""
Edge Detection & Contour Detection Module - Technique 2 of 3

Handles:
- Canny edge detection to find receipt borders
- Contour detection
- 4-point perspective transform for deskewing and cropping
- Document scanner-like functionality

This technique creates a document-scanner effect that significantly improves
OCR accuracy by normalizing receipt orientation and removing surrounding clutter.
"""

import cv2
import numpy as np


def detect_and_warp(image: np.ndarray, preprocessed: np.ndarray) -> tuple:
    """
    Detect receipt edges, find contours, and apply perspective transformation.
    
    Args:
        image: Original color image (for contour detection)
        preprocessed: Preprocessed binary image (output from preprocessor.py)
    
    Returns:
        tuple: (warped_image, contour_image)
               - warped_image: Perspective-transformed receipt image
               - contour_image: Image showing detected contours
    """
    
    # Step 1: Apply Canny Edge Detection
    # Detects edges with automatic threshold calculation
    edges = cv2.Canny(preprocessed, 50, 150)
    
    # Step 2: Dilate edges to connect nearby contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Step 3: Find Contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for visualization
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Step 4: Find the largest contour (likely the receipt)
    if len(contours) == 0:
        # If no contours found, return original image
        return image, contour_image
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Step 5: Approximate the contour to a 4-point polygon (receipt corners)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If not a 4-point polygon, try to find the bounding quadrilateral
    if len(approx) != 4:
        # Fall back to bounding rectangle if not a 4-point shape
        x, y, w, h = cv2.boundingRect(largest_contour)
        approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    
    # Step 6: Apply Perspective Transformation
    warped = perspective_transform(image, approx)
    
    return warped, contour_image


def perspective_transform(image: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to deskew and crop the receipt.
    
    Args:
        image: Original image
        contour_points: 4 corner points of the detected receipt
    
    Returns:
        Perspective-transformed (warped) image
    """
    
    # Sort points in order: top-left, top-right, bottom-right, bottom-left
    pts = contour_points.reshape(4, 2)
    rect = order_points(pts)
    
    # Calculate the width and height of the new image
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))
    
    # Define output points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Get perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply perspective transformation
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order corner points in consistent order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points
    
    Returns:
        Ordered points
    """
    
    # Sort by x-coordinate, then by y-coordinate
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Separate left and right points
    left_pts = x_sorted[:2]
    right_pts = x_sorted[2:]
    
    # Sort left points by y-coordinate
    left_pts = left_pts[np.argsort(left_pts[:, 1]), :]
    tl, bl = left_pts
    
    # Sort right points by y-coordinate
    right_pts = right_pts[np.argsort(right_pts[:, 1]), :]
    tr, br = right_pts
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_contour_outline(image: np.ndarray) -> tuple:
    """
    Get the outline contours of the receipt for visualization.
    
    Args:
        image: Preprocessed image
    
    Returns:
        tuple: (edges, contours)
    """
    
    edges = cv2.Canny(image, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return edges, contours
