"""
Annotation Module

Draws colored bounding boxes and annotations on receipt images.
Uses pytesseract data to get text region coordinates.
"""

import cv2
import numpy as np
import pytesseract


def annotate_receipt(image: np.ndarray, text_regions: list, 
                    recognized_items: set = None) -> np.ndarray:
    """
    Annotate receipt image with bounding boxes for recognized items.
    
    Args:
        image: Original receipt image (color, BGR format)
        text_regions: List of text regions from OCR (from ocr_extractor.get_text_regions)
        recognized_items: Set of recognized item names (for color coding)
    
    Returns:
        Annotated image with bounding boxes
    """
    
    if recognized_items is None:
        recognized_items = set()
    
    annotated = image.copy()
    
    for region in text_regions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        text = region['text'].lower().strip()
        
        # Determine box color based on whether item is recognized
        if any(item.lower() in text or text in item.lower() for item in recognized_items):
            # Green for recognized items
            color = (0, 255, 0)
            thickness = 2
        else:
            # Gray for unrecognized text
            color = (128, 128, 128)
            thickness = 1
        
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        
        # Add text label (only for recognized items to reduce clutter)
        if any(item.lower() in text or text in item.lower() for item in recognized_items):
            cv2.putText(annotated, text[:20], (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return annotated


def draw_bounding_boxes(image: np.ndarray, regions: list, 
                       color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes around detected text regions.
    
    Args:
        image: Input image
        regions: List of regions with x, y, w, h keys
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with drawn boxes
    """
    
    annotated = image.copy()
    
    for region in regions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    
    return annotated


def draw_highlighted_items(image: np.ndarray, items_with_coords: list) -> np.ndarray:
    """
    Draw highlighted boxes for parsed grocery items.
    
    Args:
        image: Input image
        items_with_coords: List of dicts with 'item_name', 'x', 'y', 'w', 'h'
    
    Returns:
        Annotated image
    """
    
    annotated = image.copy()
    
    for item_info in items_with_coords:
        x = item_info.get('x', 0)
        y = item_info.get('y', 0)
        w = item_info.get('w', 100)
        h = item_info.get('h', 20)
        item_name = item_info.get('item_name', 'Unknown')
        
        # Green highlight for recognized items
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add item name as label
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(annotated, item_name[:25], (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    
    return annotated


def create_annotation_legend() -> dict:
    """
    Create legend information for annotation colors.
    
    Returns:
        Dictionary with color meanings
    """
    
    return {
        'green': 'Recognized food items',
        'gray': 'Unrecognized text',
        'rgb_green': (0, 255, 0),
        'rgb_gray': (128, 128, 128)
    }


def overlay_text_on_image(image: np.ndarray, text: str, 
                         position: tuple = (10, 30),
                         font_scale: float = 0.7,
                         color: tuple = (255, 255, 255),
                         thickness: int = 1,
                         bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Overlay text on image with optional background.
    
    Args:
        image: Input image
        text: Text to overlay
        position: (x, y) position
        font_scale: Font size scale
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Background color (set to None for no background)
    
    Returns:
        Image with overlaid text
    """
    
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle if specified
    if bg_color is not None:
        cv2.rectangle(
            annotated,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1
        )
    
    # Draw text
    cv2.putText(annotated, text, (x, y), font, font_scale, color, thickness)
    
    return annotated


def create_side_by_side_comparison(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Create side-by-side comparison of two images.
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        Side-by-side comparison image
    """
    
    # Resize images to same height if needed
    height = max(original.shape[0], processed.shape[0])
    
    original_resized = cv2.resize(original, 
                                 (int(original.shape[1] * height / original.shape[0]), height))
    processed_resized = cv2.resize(processed,
                                  (int(processed.shape[1] * height / processed.shape[0]), height))
    
    # Concatenate horizontally
    comparison = np.hstack([original_resized, processed_resized])
    
    return comparison
