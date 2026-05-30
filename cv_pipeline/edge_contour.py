"""
Edge Detection & Contour Detection Module - Technique 2 of 3

Handles:
- Canny edge detection to find receipt borders
- Contour detection
- 4-point perspective transform for deskewing and cropping
- Document scanner-like functionality
"""

import cv2
import numpy as np


def detect_and_warp(image: np.ndarray, preprocessed: np.ndarray) -> tuple:
    """
    Detect receipt edges, find contours, and apply perspective transformation.

    Args:
        image: Original color image
        preprocessed: Preprocessed binary image (output from preprocessor.py)

    Returns:
        tuple: (warped_image, contour_image)
    """

    # Always work from a clean grayscale of the ORIGINAL image for edge detection.
    # The binary preprocessed image produces thousands of micro-edges that break
    # contour detection entirely.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Blur to suppress noise before Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 120)

    # Dilate to close gaps in receipt borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualization — always safe to draw on a color copy
    contour_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    # --- No contours at all: return original image ---
    if len(contours) == 0:
        return _safe_return(image), contour_image

    # Filter noise contours (< 1% of frame)
    contours = [c for c in contours if cv2.contourArea(c) > img_area * 0.01]
    if len(contours) == 0:
        return _safe_return(image), contour_image

    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)

    # --- If receipt fills >85% of the frame, perspective warp adds no value ---
    # Just crop to the bounding rect and return. This is the most common case
    # when the user photographs a flat receipt on a desk filling the whole frame.
    if largest_area / img_area > 0.85:
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Add small padding so we don't clip edge characters
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        cropped = image[y1:y2, x1:x2]
        cv2.rectangle(contour_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        return cropped, contour_image

    # --- Try to get a clean 4-point approximation for perspective warp ---
    approx = None
    for eps_factor in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
        epsilon = eps_factor * cv2.arcLength(largest_contour, True)
        candidate = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is None or len(approx) != 4:
        # Fallback: bounding rect
        x, y, w, h = cv2.boundingRect(largest_contour)
        approx = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.float32
        )
    else:
        approx = approx.reshape(4, 2).astype(np.float32)

    # Draw detected outline in blue
    approx_int = approx.astype(np.int32).reshape((-1, 1, 2))
    cv2.drawContours(contour_image, [approx_int], -1, (255, 0, 0), 3)

    # Perspective transform
    warped = perspective_transform(image, approx)

    # Safety check: if warp produced a tiny or empty image, fall back
    if warped is None or warped.shape[0] < 50 or warped.shape[1] < 50:
        return _safe_return(image), contour_image

    return warped, contour_image


def _safe_return(image: np.ndarray) -> np.ndarray:
    """Return a guaranteed color copy of the image."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def perspective_transform(image: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to deskew and crop the receipt.

    Args:
        image: Original image
        contour_points: 4 corner points (float32)

    Returns:
        Warped image, or None if transform is degenerate
    """

    try:
        pts = contour_points.reshape(4, 2).astype(np.float32)
        rect = order_points(pts)

        (tl, tr, br, bl) = rect
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))

        if max_width < 50 or max_height < 50:
            return None

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped

    except Exception:
        return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points: top-left, top-right, bottom-right, bottom-left.
    """

    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_pts = x_sorted[:2]
    right_pts = x_sorted[2:]

    left_pts = left_pts[np.argsort(left_pts[:, 1]), :]
    tl, bl = left_pts

    right_pts = right_pts[np.argsort(right_pts[:, 1]), :]
    tr, br = right_pts

    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_contour_outline(image: np.ndarray) -> tuple:
    """
    Get outline contours for visualization.

    Args:
        image: Original or grayscale image

    Returns:
        tuple: (edges, contours)
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return edges, contours