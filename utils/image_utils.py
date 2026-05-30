"""
Image Utilities Module

Provides helper functions for:
- Loading images from file paths
- Converting between PIL and OpenCV formats
- Resizing images for display
- Image validation and preprocessing
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (BGR format)
    """
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def load_image_pil(image_path: str) -> Image.Image:
    """
    Load an image as PIL Image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image object
    """
    
    return Image.open(image_path)


def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR).
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Image as numpy array (BGR format)
    """
    
    # Convert PIL to RGB numpy array
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


def cv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (BGR or grayscale) to PIL Image.

    Handles both 3-channel BGR images and single-channel grayscale/binary
    images safely. The original version crashed on grayscale images because
    it blindly applied COLOR_BGR2RGB to a 2D array.

    Args:
        cv_image: Image as numpy array (BGR or grayscale)

    Returns:
        PIL Image object
    """

    if cv_image is None:
        raise ValueError("cv_to_pil received None image")

    if len(cv_image.shape) == 2:
        # Grayscale or binary image — convert to uint8 and wrap directly
        pil_image = Image.fromarray(cv_image.astype(np.uint8))
    else:
        # Color BGR — convert to RGB for PIL/Streamlit display
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
    return pil_image


def resize_for_display(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize image to fit within max dimensions while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image
    """
    
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    if scale >= 1.0:
        return image
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def validate_image(image: np.ndarray) -> bool:
    """
    Validate if image is valid.
    
    Args:
        image: Image to validate
    
    Returns:
        True if valid, False otherwise
    """
    
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if len(image.shape) not in [2, 3]:
        return False
    return True


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with image information
    """
    
    height, width = image.shape[:2]
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_mb': (image.nbytes / 1024 / 1024)
    }


def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust image brightness.
    
    Args:
        image: Input image
        factor: Brightness factor (1.0 = original, > 1.0 = brighter, < 1.0 = darker)
    
    Returns:
        Brightness-adjusted image
    """
    
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust image contrast.
    
    Args:
        image: Input image
        factor: Contrast factor (1.0 = original, > 1.0 = higher contrast)
    
    Returns:
        Contrast-adjusted image
    """
    
    if factor == 1.0:
        return image.copy()
    
    # Adjust contrast: new_image = (image - 128) * factor + 128
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=128 * (1 - factor))
    return adjusted


def image_to_bytes(image: np.ndarray, format: str = 'JPEG') -> bytes:
    """
    Convert image to bytes for transmission/storage.
    
    Args:
        image: Input image (BGR format)
        format: Image format ('JPEG', 'PNG')
    
    Returns:
        Image as bytes
    """
    
    pil_image = cv_to_pil(image)
    bytes_io = BytesIO()
    pil_image.save(bytes_io, format=format)
    return bytes_io.getvalue()


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert bytes to image array.
    
    Args:
        image_bytes: Image as bytes
    
    Returns:
        Image as numpy array (BGR format)
    """
    
    pil_image = Image.open(BytesIO(image_bytes))
    cv_image = pil_to_cv(pil_image)
    return cv_image