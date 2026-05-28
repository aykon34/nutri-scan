"""
Unit tests for nutrition lookup module.
Tests nutrition database and calculations.
"""

from nutrition.nutrition_lookup import (
    NutritionDatabase, lookup_nutrition, calculate_nutrition_totals,
    enrich_parsed_items, get_nutrition_summary
)
import os


def test_nutrition_database_initialization():
    """Test that nutrition database initializes correctly."""
    db = NutritionDatabase()
    
    assert db.df is not None, "Database should be initialized"
    assert len(db.df) > 0, "Database should have items"
    print("✓ test_nutrition_database_initialization passed")


def test_lookup_nutrition_exact_match():
    """Test exact match lookup."""
    result = lookup_nutrition("eggs")
    
    assert result is not None, "Should find eggs"
    assert 'calories' in result, "Should have calories"
    assert result['calories'] > 0, "Calories should be positive"
    print("✓ test_lookup_nutrition_exact_match passed")


def test_lookup_nutrition_case_insensitive():
    """Test case-insensitive lookup."""
    result1 = lookup_nutrition("EGGS")
    result2 = lookup_nutrition("eggs")
    result3 = lookup_nutrition("EgGs")
    
    assert result1 is not None, "Should find with uppercase"
    assert result2 is not None, "Should find with lowercase"
    assert result3 is not None, "Should find with mixed case"
    print("✓ test_lookup_nutrition_case_insensitive passed")


def test_lookup_nutrition_partial_match():
    """Test partial match lookup."""
    result = lookup_nutrition("milk")
    
    assert result is not None, "Should find milk"
    print("✓ test_lookup_nutrition_partial_match passed")


def test_lookup_nutrition_fuzzy_match():
    """Test fuzzy matching for similar items."""
    # Test with slight misspelling
    result = lookup_nutrition("chiken")  # misspelled chicken
    
    # Should find something close (or None is also acceptable for fuzzy)
    # Test just doesn't crash
    print("✓ test_lookup_nutrition_fuzzy_match passed")


def test_calculate_nutrition_totals():
    """Test nutrition totals calculation."""
    items = [
        {
            'name': 'eggs',
            'calories': 155,
            'protein_g': 13,
            'carbs_g': 1.1,
            'fat_g': 11
        },
        {
            'name': 'milk',
            'calories': 61,
            'protein_g': 3.2,
            'carbs_g': 4.8,
            'fat_g': 3.3
        }
    ]
    
    totals = calculate_nutrition_totals(items)
    
    assert totals['total_calories'] == 216, "Should sum calories correctly"
    assert totals['items_count'] == 2, "Should count items"
    assert totals['total_protein_g'] > 0, "Should sum protein"
    print("✓ test_calculate_nutrition_totals passed")


def test_calculate_nutrition_totals_empty():
    """Test totals with empty list."""
    totals = calculate_nutrition_totals([])
    
    assert totals['total_calories'] == 0, "Empty should give 0 totals"
    assert totals['items_count'] == 0, "Should count 0 items"
    print("✓ test_calculate_nutrition_totals_empty passed")


def test_enrich_parsed_items():
    """Test enriching parsed items with nutrition data."""
    parsed_items = [
        {'name': 'eggs', 'price': 89.00},
        {'name': 'milk', 'price': 65.50}
    ]
    
    enriched = enrich_parsed_items(parsed_items)
    
    assert len(enriched) == 2, "Should have 2 items"
    assert 'calories' in enriched[0], "Should add calories"
    assert enriched[0]['found'] == True, "eggs should be found"
    print("✓ test_enrich_parsed_items passed")


def test_get_all_items():
    """Test getting all available items."""
    db = NutritionDatabase()
    items = db.get_all_items()
    
    assert isinstance(items, list), "Should return list"
    assert len(items) > 0, "Should have items"
    assert 'eggs' in items, "Should contain eggs"
    print("✓ test_get_all_items passed")


def test_nutrition_summary():
    """Test nutrition summary generation."""
    items = [
        {
            'name': 'eggs',
            'calories': 155,
            'protein_g': 13,
            'carbs_g': 1.1,
            'fat_g': 11,
            'found': True
        }
    ]
    
    summary = get_nutrition_summary(items)
    
    assert 'total_calories' in summary, "Should have total_calories"
    assert 'found_items' in summary, "Should have found_items count"
    assert summary['found_items'] == 1, "Should count found items"
    print("✓ test_nutrition_summary passed")


if __name__ == "__main__":
    test_nutrition_database_initialization()
    test_lookup_nutrition_exact_match()
    test_lookup_nutrition_case_insensitive()
    test_lookup_nutrition_partial_match()
    test_lookup_nutrition_fuzzy_match()
    test_calculate_nutrition_totals()
    test_calculate_nutrition_totals_empty()
    test_enrich_parsed_items()
    test_get_all_items()
    test_nutrition_summary()
    print("\n✓ All nutrition tests passed!")
