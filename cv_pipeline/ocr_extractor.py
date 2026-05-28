"""
OCR Text Extraction Module - Technique 3 of 3

Handles:
- Tesseract OCR integration
- Text region detection and extraction
- Bounding box coordinate collection for visualization

This is the final step in the CV pipeline that extracts actual food items
and prices from the receipt image.
"""

import pytesseract
import cv2
import numpy as np


def extract_text(image: np.ndarray) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image: Preprocessed image (can be color or grayscale)
    
    Returns:
        Raw OCR text string
    """
    
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply OCR
        text = pytesseract.image_to_string(gray)
        
        return text
    
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or not found in PATH")
        raise


def extract_text_with_config(image: np.ndarray, 
                            config: str = '--psm 6') -> str:
    """
    Extract text with custom Tesseract configuration.
    
    PSM (Page Segmentation Mode) options:
    - 0: Orientation and script detection (OSD) only
    - 1: Automatic page segmentation with OSD
    - 3: Fully automatic page segmentation (default)
    - 6: Uniform block of text
    - 11: Sparse text; find as much text as possible
    
    Args:
        image: Input image
        config: Tesseract config string
    
    Returns:
        Extracted text
    """
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    text = pytesseract.image_to_string(gray, config=config)
    return text


def get_text_regions(image: np.ndarray) -> list:
    """
    Get bounding boxes and confidence scores for detected text regions.
    
    Args:
        image: Input image
    
    Returns:
        List of dicts with keys: x, y, w, h, text, confidence
    """
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Get detailed information about text regions
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        regions = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])
            
            # Skip empty text or very low confidence
            if text and confidence > 20:
                regions.append({
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'text': text,
                    'confidence': confidence
                })
        
        return regions
    
    except Exception as e:
        print(f"Error extracting text regions: {e}")
        return []


def extract_lines(image: np.ndarray) -> list:
    """
    Extract text organized by line.
    
    Args:
        image: Input image
    
    Returns:
        List of text lines
    """
    
    text = extract_text(image)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines


def get_ocr_metadata(image: np.ndarray) -> dict:
    """
    Get detailed OCR metadata including confidence and layout.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with OCR metadata
    """
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    try:
        # Get full data
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        # Get overall metrics
        text = pytesseract.image_to_string(gray)
        
        # Calculate average confidence
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'raw_text': text,
            'num_text_regions': len([c for c in data['conf'] if int(c) > 20]),
            'avg_confidence': avg_confidence,
            'text_regions': get_text_regions(image)
        }
    
    except Exception as e:
        print(f"Error getting OCR metadata: {e}")
        return {
            'raw_text': '',
            'num_text_regions': 0,
            'avg_confidence': 0,
            'text_regions': []
        }
