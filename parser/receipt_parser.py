"""
Receipt Parser Module

Handles parsing of raw OCR text to extract:
- Food item names
- Prices
- Quantities
- Other relevant information from receipts

Uses regex patterns tailored to typical receipt formats.
"""

import re
from typing import List, Dict, Tuple


# Common patterns for receipt items
PATTERNS = {
    'item_with_price': r'([A-Za-z\s\d\(\)\/]+?)\s+(\d+(?:\.\d{2})?)\s*$',
    'qty_item_price': r'(\d+)\s*[xX]\s*([A-Za-z\s\d]+?)\s+(\d+(?:\.\d{2})?)',
    'price': r'\d+(?:\.\d{2})?',
    'quantity': r'(\d+(?:\.\d+)?)\s*(?:kg|g|ml|l|pcs?|pieces?)',
}


def parse_items(raw_ocr_text: str) -> List[Dict[str, any]]:
    """
    Parse raw OCR text to extract food items and prices.
    
    Args:
        raw_ocr_text: Raw text from OCR extraction
    
    Returns:
        List of dicts with keys: 'name', 'price', 'quantity', 'confidence'
    """
    
    items = []
    lines = raw_ocr_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Try to extract item and price
        parsed = extract_prices_and_items(line)
        if parsed:
            items.append(parsed)
    
    return items


def extract_prices_and_items(line: str) -> Dict[str, any]:
    """
    Extract item name and price from a single line.
    
    Args:
        line: Single line of receipt text
    
    Returns:
        Dict with 'name' and 'price', or None if no match
    """
    
    line = line.strip()
    if not line:
        return None
    
    # Pattern: ITEM_NAME PRICE
    # Example: EGGS 1KG 89.00
    pattern = r'([A-Za-z\s\d\(\)\/\-]+?)\s+(\d+(?:\.\d{2})?)\s*$'
    match = re.search(pattern, line)
    
    if match:
        item_name = match.group(1).strip()
        price_str = match.group(2).strip()
        
        # Filter out lines that are just numbers or timestamps
        if len(item_name) >= 2 and not item_name.isdigit():
            try:
                price = float(price_str)
                
                # Extract quantity if present
                quantity_match = re.search(PATTERNS['quantity'], item_name)
                quantity = None
                if quantity_match:
                    quantity = quantity_match.group(1)
                
                return {
                    'name': item_name,
                    'price': price,
                    'quantity': quantity,
                    'confidence': 0.8  # Default confidence
                }
            except ValueError:
                pass
    
    return None


def clean_item_name(name: str) -> str:
    """
    Clean and normalize item name.
    
    Args:
        name: Raw item name from OCR
    
    Returns:
        Cleaned item name
    """
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Remove common receipt artifacts
    name = re.sub(r'\*+', '', name)
    name = re.sub(r'#+', '', name)
    name = re.sub(r'\.{2,}', '', name)
    
    # Convert to title case
    name = name.title()
    
    # Remove quantity units from the name
    name = re.sub(r'\s*(?:kg|g|ml|l|pcs?|pieces?|oz|lb|lbs)', '', name, flags=re.IGNORECASE)
    
    return name.strip()


def extract_quantity_and_unit(text: str) -> Tuple[float, str]:
    """
    Extract quantity and unit from text.
    
    Args:
        text: Text containing quantity and unit
    
    Returns:
        Tuple of (quantity, unit)
    """
    
    # Pattern: number UNIT
    match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', text)
    
    if match:
        quantity = float(match.group(1))
        unit = match.group(2).lower()
        return quantity, unit
    
    return None, None


def validate_price(price_str: str) -> bool:
    """
    Validate if a string represents a valid price.
    
    Args:
        price_str: Price string to validate
    
    Returns:
        True if valid price format
    """
    
    pattern = r'^\d+(?:\.\d{1,2})?$'
    return bool(re.match(pattern, price_str.strip()))


def extract_total_price(text: str) -> float:
    """
    Extract total price from receipt text.
    
    Args:
        text: Receipt text
    
    Returns:
        Total price as float, or None
    """
    
    # Look for patterns like "TOTAL: 123.45" or "TOTAL 123.45"
    patterns = [
        r'TOTAL\s*:?\s*(\d+(?:\.\d{2})?)',
        r'SUBTOTAL\s*:?\s*(\d+(?:\.\d{2})?)',
        r'AMOUNT\s*:?\s*(\d+(?:\.\d{2})?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    
    return None


def extract_date(text: str) -> str:
    """
    Extract date from receipt text.
    
    Args:
        text: Receipt text
    
    Returns:
        Date string if found, None otherwise
    """
    
    # Common date patterns: MM/DD/YY, MM/DD/YYYY, DD/MM/YY, etc.
    patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # MM-DD-YYYY or variations
        r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2,4}',  # DD Month YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None


def extract_time(text: str) -> str:
    """
    Extract time from receipt text.
    
    Args:
        text: Receipt text
    
    Returns:
        Time string if found, None otherwise
    """
    
    # Pattern: HH:MM or HH:MM:SS
    pattern = r'\d{1,2}:\d{2}(?::\d{2})?'
    match = re.search(pattern, text)
    
    if match:
        return match.group(0)
    
    return None


def parse_receipt_header(text: str) -> Dict[str, str]:
    """
    Extract header information from receipt.
    
    Args:
        text: Receipt text
    
    Returns:
        Dict with store name, date, time, etc.
    """
    
    header_info = {}
    
    # Extract date
    date = extract_date(text)
    if date:
        header_info['date'] = date
    
    # Extract time
    time = extract_time(text)
    if time:
        header_info['time'] = time
    
    # Extract total
    total = extract_total_price(text)
    if total:
        header_info['total'] = total
    
    # First line is usually store name
    lines = text.split('\n')
    if lines:
        header_info['store_name'] = lines[0].strip()
    
    return header_info


def merge_split_items(items: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Merge items that were split across multiple lines.
    
    Args:
        items: List of parsed items
    
    Returns:
        Merged items list
    """
    
    if not items:
        return items
    
    merged = []
    current_item = None
    
    for item in items:
        if current_item and item['name'].lower().startswith(current_item['name'].lower()):
            # Same item continued on next line
            current_item['name'] += ' ' + item['name']
        else:
            if current_item:
                merged.append(current_item)
            current_item = item.copy()
    
    if current_item:
        merged.append(current_item)
    
    return merged
