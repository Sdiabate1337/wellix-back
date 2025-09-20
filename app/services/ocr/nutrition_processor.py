"""
Nutrition-specific OCR processor for enhanced text extraction.
Post-processes OCR results with domain knowledge.

Architecture Pattern: Strategy + Domain-Driven Design + Pipeline
"""

import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import structlog

from app.services.ocr.interfaces import IOCRProcessor, OCRResult


logger = structlog.get_logger(__name__)


class NutritionOCRProcessor(IOCRProcessor):
    """
    Nutrition-specialized OCR post-processor.
    
    Features:
    - Nutrition fact table detection
    - Unit normalization (g, mg, kcal, etc.)
    - Ingredient list parsing
    - Allergen detection
    - Serving size extraction
    """
    
    def __init__(self):
        self.nutrition_keywords = self._load_nutrition_keywords()
        self.unit_patterns = self._compile_unit_patterns()
        self.allergen_keywords = self._load_allergen_keywords()
    
    def _load_nutrition_keywords(self) -> Dict[str, List[str]]:
        """Load nutrition-related keywords by category."""
        return {
            "macronutrients": [
                "calories", "kcal", "energy", "protein", "fat", "carbohydrates", 
                "carbs", "fiber", "fibre", "sugar", "sodium", "salt",
                "calories", "énergie", "protéines", "matières grasses", 
                "glucides", "fibres", "sucres", "sodium", "sel"
            ],
            "vitamins": [
                "vitamin a", "vitamin c", "vitamin d", "vitamin e", "vitamin k",
                "thiamin", "riboflavin", "niacin", "folate", "b12", "b6",
                "vitamine a", "vitamine c", "vitamine d", "vitamine e"
            ],
            "minerals": [
                "calcium", "iron", "magnesium", "phosphorus", "potassium", 
                "zinc", "copper", "manganese", "selenium",
                "calcium", "fer", "magnésium", "phosphore", "potassium"
            ],
            "sections": [
                "nutrition facts", "nutrition information", "nutritional info",
                "ingredients", "contains", "allergens", "serving size",
                "valeurs nutritionnelles", "informations nutritionnelles",
                "ingrédients", "contient", "allergènes", "portion"
            ]
        }
    
    def _compile_unit_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for unit detection."""
        patterns = [
            # Standard units with values
            r'(\d+(?:\.\d+)?)\s*(g|mg|kg|kcal|cal|kj|ml|l|%|oz|lb)\b',
            # Spelled out units
            r'(\d+(?:\.\d+)?)\s*(grams?|milligrams?|kilograms?|calories?|percent|ounces?|pounds?)\b',
            # French units
            r'(\d+(?:\.\d+)?)\s*(grammes?|milligrammes?|kilogrammes?|pourcentage?)\b',
            # Serving information
            r'(\d+(?:\.\d+)?)\s*(servings?|portions?|pieces?|pièces?)\b'
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_allergen_keywords(self) -> List[str]:
        """Load common allergen keywords."""
        return [
            "milk", "eggs", "fish", "shellfish", "tree nuts", "peanuts", 
            "wheat", "soybeans", "soy", "sesame", "gluten", "sulfites",
            "lait", "œufs", "poisson", "crustacés", "noix", "arachides",
            "blé", "soja", "sésame", "gluten", "sulfites"
        ]
    
    async def enhance_nutrition_extraction(self, ocr_result: OCRResult) -> OCRResult:
        """
        Enhance OCR result with comprehensive nutrition processing.
        
        Processing pipeline:
        1. Detect nutrition keywords
        2. Extract numerical values with units
        3. Parse ingredient lists
        4. Identify allergens
        5. Structure nutrition facts
        """
        # Run all enhancement processes
        enhanced_result = ocr_result
        
        # Detect nutrition keywords
        enhanced_result.nutrition_keywords = await self.detect_nutrition_keywords(
            enhanced_result.raw_text
        )
        
        # Extract numerical values
        enhanced_result.detected_numbers = await self.extract_numerical_values(
            enhanced_result.raw_text
        )
        
        # Parse ingredient sections
        enhanced_result.ingredient_sections = await self._parse_ingredient_sections(
            enhanced_result.raw_text
        )
        
        # Add structured nutrition data
        nutrition_data = await self._structure_nutrition_facts(enhanced_result)
        
        # Store in metadata (extend OCRResult if needed)
        if not hasattr(enhanced_result, 'structured_data'):
            # Add to a metadata dict for now
            pass
        
        logger.info(
            "OCR enhancement completed",
            keywords_found=len(enhanced_result.nutrition_keywords),
            numbers_found=len(enhanced_result.detected_numbers),
            ingredients_found=len(enhanced_result.ingredient_sections)
        )
        
        return enhanced_result
    
    async def detect_nutrition_keywords(self, text: str) -> List[str]:
        """Detect nutrition-related keywords in text."""
        text_lower = text.lower()
        found_keywords = []
        
        for category, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_keywords))
    
    async def extract_numerical_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical values with context and units."""
        detected_values = []
        
        for pattern in self.unit_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Get surrounding context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                # Classify the type of measurement
                measurement_type = self._classify_measurement(context, unit)
                
                detected_values.append({
                    "value": value,
                    "unit": unit,
                    "context": context,
                    "type": measurement_type,
                    "position": {"start": match.start(), "end": match.end()}
                })
        
        return detected_values
    
    def _classify_measurement(self, context: str, unit: str) -> str:
        """Classify what type of nutritional measurement this is."""
        context_lower = context.lower()
        
        # Macronutrient detection
        if any(word in context_lower for word in ["calorie", "kcal", "energy", "énergie"]):
            return "calories"
        elif any(word in context_lower for word in ["protein", "protéine"]):
            return "protein"
        elif any(word in context_lower for word in ["fat", "lipid", "matière grasse"]):
            return "fat"
        elif any(word in context_lower for word in ["carb", "glucide", "carbohydrate"]):
            return "carbohydrates"
        elif any(word in context_lower for word in ["fiber", "fibre"]):
            return "fiber"
        elif any(word in context_lower for word in ["sugar", "sucre"]):
            return "sugar"
        elif any(word in context_lower for word in ["sodium", "salt", "sel"]):
            return "sodium"
        
        # Vitamin/mineral detection
        elif any(word in context_lower for word in ["vitamin", "vitamine"]):
            return "vitamin"
        elif any(word in context_lower for word in ["calcium", "iron", "fer", "magnesium"]):
            return "mineral"
        
        # Serving size
        elif any(word in context_lower for word in ["serving", "portion", "container"]):
            return "serving_size"
        
        # Default based on unit
        elif unit in ["g", "mg", "kg", "gram", "milligram"]:
            return "weight"
        elif unit in ["ml", "l", "liter", "litre"]:
            return "volume"
        elif unit in ["%", "percent", "pourcentage"]:
            return "percentage"
        
        return "unknown"
    
    async def _parse_ingredient_sections(self, text: str) -> List[str]:
        """Parse and extract ingredient list sections."""
        ingredient_sections = []
        text_lower = text.lower()
        
        # Common ingredient section markers
        markers = [
            "ingredients:", "ingrédients:", "composition:",
            "contains:", "contient:", "made with:",
            "ingredients list", "liste d'ingrédients"
        ]
        
        for marker in markers:
            marker_pos = text_lower.find(marker.lower())
            if marker_pos != -1:
                # Find the actual position in original text
                start_pos = marker_pos + len(marker)
                
                # Extract until next major section or reasonable cutoff
                remaining_text = text[start_pos:].strip()
                
                # Look for end markers
                end_markers = [
                    "nutrition facts", "nutritional information",
                    "allergen", "storage", "directions",
                    "valeurs nutritionnelles", "allergène", "conservation"
                ]
                
                end_pos = len(remaining_text)
                for end_marker in end_markers:
                    marker_pos = remaining_text.lower().find(end_marker.lower())
                    if marker_pos != -1:
                        end_pos = min(end_pos, marker_pos)
                
                # Extract ingredient text
                ingredient_text = remaining_text[:end_pos].strip()
                
                # Clean up the text
                ingredient_text = self._clean_ingredient_text(ingredient_text)
                
                if len(ingredient_text) > 10:  # Only substantial text
                    ingredient_sections.append(ingredient_text)
        
        return ingredient_sections
    
    def _clean_ingredient_text(self, text: str) -> str:
        """Clean up ingredient text."""
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[|]{2,}', '', text)  # Multiple pipes
        text = re.sub(r'[-]{3,}', '', text)  # Multiple dashes
        
        # Remove trailing punctuation that's likely OCR error
        text = text.strip('.,;:')
        
        return text.strip()
    
    async def _structure_nutrition_facts(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Structure detected nutrition facts into organized data."""
        structured_data = {
            "macronutrients": {},
            "vitamins": {},
            "minerals": {},
            "serving_info": {},
            "allergens": [],
            "ingredients": ocr_result.ingredient_sections
        }
        
        # Process detected numbers to extract nutrition facts
        for number_data in ocr_result.detected_numbers:
            measurement_type = number_data["type"]
            value = number_data["value"]
            unit = number_data["unit"]
            
            if measurement_type in ["calories", "protein", "fat", "carbohydrates", "fiber", "sugar", "sodium"]:
                structured_data["macronutrients"][measurement_type] = {
                    "value": value,
                    "unit": unit,
                    "context": number_data["context"]
                }
            elif measurement_type == "vitamin":
                # Try to identify specific vitamin from context
                context = number_data["context"].lower()
                for vitamin in ["vitamin a", "vitamin c", "vitamin d", "vitamin e"]:
                    if vitamin in context:
                        structured_data["vitamins"][vitamin] = {
                            "value": value,
                            "unit": unit
                        }
                        break
            elif measurement_type == "mineral":
                # Try to identify specific mineral
                context = number_data["context"].lower()
                for mineral in ["calcium", "iron", "magnesium", "potassium"]:
                    if mineral in context:
                        structured_data["minerals"][mineral] = {
                            "value": value,
                            "unit": unit
                        }
                        break
            elif measurement_type == "serving_size":
                structured_data["serving_info"]["size"] = {
                    "value": value,
                    "unit": unit
                }
        
        # Detect allergens in text
        text_lower = ocr_result.raw_text.lower()
        for allergen in self.allergen_keywords:
            if allergen.lower() in text_lower:
                structured_data["allergens"].append(allergen)
        
        return structured_data