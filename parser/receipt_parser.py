"""
Receipt Parser Module

Handles parsing of raw OCR text to extract:
- Food item names
- Prices
- Quantities
- Other relevant information from receipts
"""

import re
from typing import List, Dict, Tuple


# Keywords that identify non-food summary lines — never treat as items
SKIP_KEYWORDS = {
    'total', 'subtotal', 'sub-total', 'sub total',
    'tax', 'vat', 'gst',
    'cash', 'change', 'due', 'paid',
    'receipt', 'invoice', 'order',
    'thank', 'you', 'welcome',
    'date', 'time', 'server', 'table',
    'discount', 'savings', 'coupon',
    'amount', 'balance', 'payment',
    'tip', 'gratuity', 'signature',
}

QUANTITY_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:kg|g|ml|l|pcs?|pieces?|oz|lb|lbs)',
    re.IGNORECASE
)


def _is_skip_line(line: str) -> bool:
    """Return True if this line is a receipt summary/header, not a food item."""
    lower = line.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


def parse_items(raw_ocr_text: str) -> List[Dict[str, any]]:
    """
    Parse raw OCR text to extract food items and prices.

    Handles common receipt formats:
      - "APPLE 1.00"
      - "2 APPLE 1.00"        (leading quantity)
      - "2 x APPLE 1.00"      (quantity with x)
      - "1 LEMON | 0.60"      (pipe separator)
      - "BURGER DELUXE $14.99" (dollar sign prefix)

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
        if _is_skip_line(line):
            continue

        parsed = _extract_item(line)
        if parsed:
            items.append(parsed)

    return items


def _extract_item(line: str) -> Dict[str, any]:
    """
    Extract item name and price from a single receipt line.

    Handles these formats (with optional leading quantity):
      [QTY] NAME [|] [$]PRICE

    Args:
        line: Single receipt line

    Returns:
        Dict with name/price/quantity/confidence, or None
    """

    line = line.strip()
    if not line:
        return None

    # Normalise separators: replace pipe | with space
    line = line.replace('|', ' ')
    # Collapse multiple spaces
    line = re.sub(r' {2,}', ' ', line)

    # --- Strip optional leading quantity: "2 APPLE ..." or "2 x APPLE ..." ---
    leading_qty = None
    qty_match = re.match(r'^(\d+)\s*[xX]?\s+', line)
    if qty_match:
        # Only strip it if what follows starts with a letter (it's a count, not a price)
        remainder = line[qty_match.end():]
        if remainder and remainder[0].isalpha():
            leading_qty = qty_match.group(1)
            line = remainder

    # --- Main pattern: NAME followed by optional $ then a number at end of line ---
    # Name: letters, digits, spaces, common punctuation (but NOT a bare number)
    pattern = re.compile(
        r'^([A-Za-z][A-Za-z0-9\s\(\)\/\-\%\.&\'\"]*?)'   # item name (must start with letter)
        r'\s+\$?'                                           # separator + optional $
        r'(\d{1,6}(?:[.,]\d{1,2})?)'                       # price (with . or , decimal)
        r'\s*$'
    )
    match = pattern.match(line)

    if not match:
        return None

    item_name = match.group(1).strip()
    price_str = match.group(2).replace(',', '.')  # handle comma decimals

    # Must have at least 2 letters to be a real item name
    if len(re.findall(r'[A-Za-z]', item_name)) < 2:
        return None

    # Skip summary lines that slipped through
    if _is_skip_line(item_name):
        return None

    try:
        price = float(price_str)
    except ValueError:
        return None

    # Reject implausibly large prices (likely a barcode or date)
    if price > 100000:
        return None

    # Extract inline quantity if present in name (e.g. "APPLE 1KG")
    qty_in_name = QUANTITY_PATTERN.search(item_name)
    quantity = leading_qty or (qty_in_name.group(1) if qty_in_name else None)

    return {
        'name': item_name,
        'price': price,
        'quantity': quantity,
        'confidence': 0.85,
    }


# Keep these public functions so the rest of the app doesn't break


def extract_prices_and_items(line: str) -> Dict[str, any]:
    """Public alias used by other modules."""
    return _extract_item(line)


def clean_item_name(name: str) -> str:
    """Clean and normalise item name."""
    name = ' '.join(name.split())
    name = re.sub(r'\*+', '', name)
    name = re.sub(r'#+', '', name)
    name = re.sub(r'\.{2,}', '', name)
    name = name.title()
    # CRITICAL FIX: units must be preceded by a digit or space+digit and followed
    # by a word boundary, so we never strip letters from inside food names.
    # Bad: r"\s*(?:kg|g|ml|l|...)" matched "l" inside "Apple", "Milk", "Lemon" etc.
    name = re.sub(r'(?<=\d)\s*(?:kg|g|ml|l|pcs?|pieces?|oz|lbs?)', '', name, flags=re.IGNORECASE)
    return name.strip()


def extract_quantity_and_unit(text: str) -> Tuple[float, str]:
    """Extract quantity and unit from text."""
    match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', text)
    if match:
        return float(match.group(1)), match.group(2).lower()
    return None, None


def validate_price(price_str: str) -> bool:
    """Validate if a string represents a valid price."""
    return bool(re.match(r'^\d+(?:\.\d{1,2})?$', price_str.strip()))


def extract_total_price(text: str) -> float:
    """Extract total price from receipt text."""
    patterns = [
        r'TOTAL\s*:?\s*\$?(\d+(?:\.\d{2})?)',
        r'SUBTOTAL\s*:?\s*\$?(\d+(?:\.\d{2})?)',
        r'AMOUNT\s*:?\s*\$?(\d+(?:\.\d{2})?)',
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
    """Extract date from receipt text."""
    patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2,4}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def extract_time(text: str) -> str:
    """Extract time from receipt text."""
    match = re.search(r'\d{1,2}:\d{2}(?::\d{2})?', text)
    return match.group(0) if match else None


def parse_receipt_header(text: str) -> Dict[str, str]:
    """Extract header information from receipt."""
    header_info = {}
    date = extract_date(text)
    if date:
        header_info['date'] = date
    time = extract_time(text)
    if time:
        header_info['time'] = time
    total = extract_total_price(text)
    if total:
        header_info['total'] = total
    lines = text.split('\n')
    if lines:
        header_info['store_name'] = lines[0].strip()
    return header_info


def merge_split_items(items: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """Merge items that were split across multiple lines."""
    if not items:
        return items
    merged = []
    current_item = None
    for item in items:
        if current_item and item['name'].lower().startswith(current_item['name'].lower()):
            current_item['name'] += ' ' + item['name']
        else:
            if current_item:
                merged.append(current_item)
            current_item = item.copy()
    if current_item:
        merged.append(current_item)
    return merged