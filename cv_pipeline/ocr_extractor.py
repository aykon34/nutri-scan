"""
OCR Text Extraction Module - Technique 3 of 3

Handles:
- Tesseract OCR integration
- Image enhancement before OCR (CLAHE, sharpening, upscaling)
- Text region detection and extraction
- Bounding box coordinate collection for visualization
"""

import pytesseract
import cv2
import numpy as np


def _enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhance a grayscale image so Tesseract can read it reliably.

    Steps:
      1. Upscale if too small (Tesseract needs ~300 DPI equivalent)
      2. CLAHE contrast normalisation (fixes flat/washed-out receipt scans)
      3. Unsharp-mask sharpening (crisp character edges)
      4. Denoise

    Args:
        image: Single-channel (grayscale) uint8 array

    Returns:
        Enhanced single-channel uint8 array
    """

    # --- 1. Upscale small images ---
    h, w = image.shape[:2]
    if w < 1200:
        scale = 1200 / w
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # --- 2. CLAHE contrast enhancement ---
    # Adapts locally so both bright and dark regions of the receipt
    # get readable contrast, even under uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # --- 3. Unsharp mask sharpening ---
    # Makes character edges crisp so Tesseract doesn't confuse similar glyphs.
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # --- 4. Denoise ---
    image = cv2.fastNlMeansDenoising(image, h=10,
                                     templateWindowSize=7,
                                     searchWindowSize=21)

    return image


def extract_text(image: np.ndarray) -> str:
    """
    Extract text from image using Tesseract OCR.

    Args:
        image: Input image (color or grayscale)

    Returns:
        Raw OCR text string
    """

    try:
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        enhanced = _enhance_for_ocr(gray)

        # PSM 6  = single uniform block of text  — best for receipts
        # OEM 3  = LSTM engine (most accurate modern engine)
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(enhanced, config=config)

        return text

    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or not found in PATH")
        raise


def extract_text_with_config(image: np.ndarray,
                             config: str = '--psm 6 --oem 3') -> str:
    """
    Extract text with custom Tesseract configuration.

    PSM modes useful for receipts:
    - 6: Uniform block of text (recommended default)
    - 4: Single column of variable-size text
    - 11: Sparse text — finds as much text as possible in any order

    Args:
        image: Input image
        config: Tesseract config string

    Returns:
        Extracted text
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    enhanced = _enhance_for_ocr(gray)
    text = pytesseract.image_to_string(enhanced, config=config)
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
        gray = image.copy()

    enhanced = _enhance_for_ocr(gray)

    try:
        config = '--psm 6 --oem 3'
        data = pytesseract.image_to_data(enhanced,
                                         output_type=pytesseract.Output.DICT,
                                         config=config)

        regions = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])

            # Keep anything Tesseract has even mild confidence in
            if text and confidence > 10:
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
    Extract text organised by line.

    Args:
        image: Input image

    Returns:
        List of non-empty text lines
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
        gray = image.copy()

    enhanced = _enhance_for_ocr(gray)

    try:
        config = '--psm 6 --oem 3'
        data = pytesseract.image_to_data(enhanced,
                                         output_type=pytesseract.Output.DICT,
                                         config=config)
        text = pytesseract.image_to_string(enhanced, config=config)

        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        regions = []
        for i in range(len(data['text'])):
            t = data['text'][i].strip()
            conf = int(data['conf'][i])
            if t and conf > 10:
                regions.append({
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'text': t,
                    'confidence': conf
                })

        return {
            'raw_text': text,
            'num_text_regions': len(regions),
            'avg_confidence': avg_confidence,
            'text_regions': regions
        }

    except Exception as e:
        print(f"Error getting OCR metadata: {e}")
        return {
            'raw_text': '',
            'num_text_regions': 0,
            'avg_confidence': 0.0,
            'text_regions': []
        }