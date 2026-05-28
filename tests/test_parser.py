"""
Unit tests for receipt parser module.
Tests regex parsing and item extraction functionality.
"""

from parser.receipt_parser import (
    parse_items, extract_prices_and_items, clean_item_name,
    validate_price, extract_quantity_and_unit
)


def test_extract_prices_and_items_valid():
    """Test extracting item name and price from valid line."""
    line = "EGGS 1KG 89.00"
    result = extract_prices_and_items(line)
    
    assert result is not None, "Should extract from valid line"
    assert 'name' in result, "Result should have 'name' key"
    assert 'price' in result, "Result should have 'price' key"
    assert result['price'] == 89.00, "Price should be 89.00"
    assert 'eggs' in result['name'].lower(), "Should contain 'eggs'"
    print("✓ test_extract_prices_and_items_valid passed")


def test_extract_prices_and_items_no_match():
    """Test that non-matching lines return None."""
    line = "Random text without price"
    result = extract_prices_and_items(line)
    
    assert result is None, "Should return None for invalid format"
    print("✓ test_extract_prices_and_items_no_match passed")


def test_parse_items_multiline():
    """Test parsing multiple items from multiline text."""
    text = """EGGS 1KG 89.00
MILK 1L 65.50
BREAD 400G 45.00"""
    
    items = parse_items(text)
    
    assert len(items) >= 3, "Should extract at least 3 items"
    prices = [item['price'] for item in items]
    assert 89.00 in prices, "Should find EGGS price"
    assert 65.50 in prices, "Should find MILK price"
    assert 45.00 in prices, "Should find BREAD price"
    print("✓ test_parse_items_multiline passed")


def test_clean_item_name():
    """Test item name cleaning."""
    name = "  EGGS 1KG  "
    cleaned = clean_item_name(name)
    
    assert "kg" not in cleaned.lower(), "Should remove unit"
    assert cleaned == cleaned.strip(), "Should remove extra spaces"
    print("✓ test_clean_item_name passed")


def test_validate_price():
    """Test price validation."""
    assert validate_price("89.00") == True, "Should validate valid price"
    assert validate_price("45.5") == True, "Should validate price with 1 decimal"
    assert validate_price("100") == True, "Should validate whole number"
    assert validate_price("abc") == False, "Should reject non-numeric"
    assert validate_price("89.999") == False, "Should reject price with > 2 decimals"
    print("✓ test_validate_price passed")


def test_extract_quantity_and_unit():
    """Test quantity and unit extraction."""
    quantity, unit = extract_quantity_and_unit("1.5 kg")
    
    assert quantity == 1.5, "Should extract quantity"
    assert unit == "kg", "Should extract unit"
    print("✓ test_extract_quantity_and_unit passed")


def test_parse_items_empty_text():
    """Test parsing empty text."""
    items = parse_items("")
    
    assert isinstance(items, list), "Should return list"
    assert len(items) == 0, "Should return empty list for empty input"
    print("✓ test_parse_items_empty_text passed")


def test_extract_prices_and_items_various_formats():
    """Test extraction from various receipt line formats."""
    test_cases = [
        "CHICKEN 500G 180.00",
        "RICE 2KG 250.50",
        "APPLE 1KG 120.75",
    ]
    
    for line in test_cases:
        result = extract_prices_and_items(line)
        assert result is not None, f"Should extract from: {line}"
        assert result['price'] > 0, f"Price should be positive for: {line}"
    
    print("✓ test_extract_prices_and_items_various_formats passed")


if __name__ == "__main__":
    test_extract_prices_and_items_valid()
    test_extract_prices_and_items_no_match()
    test_parse_items_multiline()
    test_clean_item_name()
    test_validate_price()
    test_extract_quantity_and_unit()
    test_parse_items_empty_text()
    test_extract_prices_and_items_various_formats()
    print("\n✓ All parser tests passed!")
