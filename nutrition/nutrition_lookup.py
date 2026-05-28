"""
Nutrition Lookup Module

Handles keyword-to-nutrient matching against local nutrition database.
Maps food item names from receipts to nutritional data.
"""

import os
import pandas as pd
from typing import Dict, List, Optional
import difflib


class NutritionDatabase:
    """Nutrition database manager."""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize nutrition database.
        
        Args:
            csv_path: Path to nutrition_data.csv (uses default if not provided)
        """
        
        if csv_path is None:
            # Default path relative to this file
            csv_path = os.path.join(os.path.dirname(__file__), 'nutrition_data.csv')
        
        try:
            self.df = pd.read_csv(csv_path)
            self.df['item_keyword'] = self.df['item_keyword'].str.lower().str.strip()
        except FileNotFoundError:
            print(f"Warning: Nutrition database not found at {csv_path}")
            self.df = pd.DataFrame(columns=['item_keyword', 'calories', 'protein_g', 'carbs_g', 'fat_g'])
    
    def lookup(self, item_name: str) -> Optional[Dict]:
        """
        Look up nutrition data for an item.
        
        Args:
            item_name: Name of the food item
        
        Returns:
            Dict with nutrition info, or None if not found
        """
        
        item_name = item_name.lower().strip()
        
        # Exact match first
        matches = self.df[self.df['item_keyword'] == item_name]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        
        # Partial/substring match
        matches = self.df[self.df['item_keyword'].str.contains(item_name, na=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        
        # Fuzzy match using difflib
        keywords = self.df['item_keyword'].tolist()
        closest = difflib.get_close_matches(item_name, keywords, n=1, cutoff=0.6)
        
        if closest:
            matches = self.df[self.df['item_keyword'] == closest[0]]
            if not matches.empty:
                return matches.iloc[0].to_dict()
        
        return None
    
    def get_all_items(self) -> List[str]:
        """
        Get list of all available items.
        
        Returns:
            List of item keywords
        """
        
        return self.df['item_keyword'].tolist()
    
    def add_item(self, keyword: str, calories: float, protein: float, 
                carbs: float, fat: float) -> None:
        """
        Add a new item to the database.
        
        Args:
            keyword: Item keyword
            calories: Calories per serving
            protein: Protein in grams
            carbs: Carbs in grams
            fat: Fat in grams
        """
        
        new_row = pd.DataFrame({
            'item_keyword': [keyword.lower().strip()],
            'calories': [calories],
            'protein_g': [protein],
            'carbs_g': [carbs],
            'fat_g': [fat]
        })
        
        self.df = pd.concat([self.df, new_row], ignore_index=True)


# Global database instance
_db = None


def get_database() -> NutritionDatabase:
    """Get or create global nutrition database."""
    global _db
    if _db is None:
        _db = NutritionDatabase()
    return _db


def lookup_nutrition(item_name: str) -> Optional[Dict]:
    """
    Look up nutrition data for an item.
    
    Args:
        item_name: Name of food item
    
    Returns:
        Dict with nutrition info or None
    """
    
    db = get_database()
    return db.lookup(item_name)


def get_all_items() -> List[str]:
    """
    Get all available items in database.
    
    Returns:
        List of item keywords
    """
    
    db = get_database()
    return db.get_all_items()


def calculate_nutrition_totals(items_with_nutrition: List[Dict]) -> Dict:
    """
    Calculate total nutrition from list of items.
    
    Args:
        items_with_nutrition: List of dicts with 'calories', 'protein_g', 'carbs_g', 'fat_g'
    
    Returns:
        Dict with totals
    """
    
    totals = {
        'total_calories': 0,
        'total_protein_g': 0,
        'total_carbs_g': 0,
        'total_fat_g': 0,
        'items_count': len(items_with_nutrition)
    }
    
    for item in items_with_nutrition:
        totals['total_calories'] += item.get('calories', 0)
        totals['total_protein_g'] += item.get('protein_g', 0)
        totals['total_carbs_g'] += item.get('carbs_g', 0)
        totals['total_fat_g'] += item.get('fat_g', 0)
    
    return totals


def enrich_parsed_items(parsed_items: List[Dict]) -> List[Dict]:
    """
    Enrich parsed items with nutrition data.
    
    Args:
        parsed_items: List of parsed items from receipt parser
    
    Returns:
        List of items with nutrition data added
    """
    
    enriched = []
    
    for item in parsed_items:
        enriched_item = item.copy()
        
        # Try to look up nutrition data
        nutrition_data = lookup_nutrition(item.get('name', ''))
        
        if nutrition_data:
            enriched_item['calories'] = nutrition_data.get('calories', 0)
            enriched_item['protein_g'] = nutrition_data.get('protein_g', 0)
            enriched_item['carbs_g'] = nutrition_data.get('carbs_g', 0)
            enriched_item['fat_g'] = nutrition_data.get('fat_g', 0)
            enriched_item['found'] = True
        else:
            enriched_item['calories'] = 0
            enriched_item['protein_g'] = 0
            enriched_item['carbs_g'] = 0
            enriched_item['fat_g'] = 0
            enriched_item['found'] = False
        
        enriched.append(enriched_item)
    
    return enriched


def get_nutrition_summary(items_with_nutrition: List[Dict]) -> Dict:
    """
    Get comprehensive nutrition summary.
    
    Args:
        items_with_nutrition: Items with nutrition data
    
    Returns:
        Detailed summary dict
    """
    
    totals = calculate_nutrition_totals(items_with_nutrition)
    
    # Calculate percentages (rough macronutrient percentages)
    total_kcal_from_macros = (
        totals['total_protein_g'] * 4 +  # 4 kcal per gram protein
        totals['total_carbs_g'] * 4 +    # 4 kcal per gram carbs
        totals['total_fat_g'] * 9        # 9 kcal per gram fat
    )
    
    summary = {
        **totals,
        'found_items': sum(1 for item in items_with_nutrition if item.get('found', False)),
        'not_found_items': sum(1 for item in items_with_nutrition if not item.get('found', False))
    }
    
    if total_kcal_from_macros > 0:
        summary['protein_percentage'] = (totals['total_protein_g'] * 4 / total_kcal_from_macros) * 100
        summary['carbs_percentage'] = (totals['total_carbs_g'] * 4 / total_kcal_from_macros) * 100
        summary['fat_percentage'] = (totals['total_fat_g'] * 9 / total_kcal_from_macros) * 100
    
    return summary
