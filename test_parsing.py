#!/usr/bin/env python3
"""
Test direct du parsing des valeurs nutritionnelles
"""

# Test data identique à ce que nous envoyons
nutrition_data = {
    "product_name": "Nutella",
    "brand": "Ferrero", 
    "serving_size": "100g",
    "nutrition": {
        "energy_kcal": 539,
        "carbohydrates": 57.5,
        "sugars": 56.3,
        "fat": 30.9,
        "saturated_fat": 10.6,
        "protein": 6.3,
        "fiber": 3.0,
        "sodium": 40
    },
    "ingredients": ["sucre", "huile de palme", "noisettes"],
    "allergens": ["noisettes", "lait", "soja"]
}

# Logique identique à celle de l'endpoint
nutrition_details = nutrition_data.get("nutrition", nutrition_data)
print(f"nutrition_details: {nutrition_details}")

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Extract numeric part from strings like "75g" or "450 kcal"
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', value)
            return float(match.group(1)) if match else default
        return default
    except Exception as e:
        print(f"safe_float error for value {value}: {e}")
        return default

# Test des extractions
print(f"Extraction carbohydrates: {nutrition_details.get('carbohydrates', 0)} -> {safe_float(nutrition_details.get('carbohydrates', 0))}")
print(f"Extraction sugars: {nutrition_details.get('sugars', 0)} -> {safe_float(nutrition_details.get('sugars', 0))}")
print(f"Extraction energy_kcal: {nutrition_details.get('energy_kcal', 0)} -> {safe_float(nutrition_details.get('energy_kcal', 0))}")
print(f"Extraction protein: {nutrition_details.get('protein', 0)} -> {safe_float(nutrition_details.get('protein', 0))}")

# Test avec structure alternative (sugar vs sugars)
print(f"Extraction sugar: {nutrition_details.get('sugar', 0)} -> {safe_float(nutrition_details.get('sugar', 0))}")
print(f"Extraction sugars fallback: {safe_float(nutrition_details.get('sugars', nutrition_details.get('sugar', 0)))}")