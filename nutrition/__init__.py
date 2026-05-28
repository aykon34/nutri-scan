"""
Nutrition module for NutriScan.
Contains nutrition lookup and analysis logic.
"""

from .nutrition_lookup import lookup_nutrition, get_all_items, calculate_nutrition_totals

__all__ = ['lookup_nutrition', 'get_all_items', 'calculate_nutrition_totals']
