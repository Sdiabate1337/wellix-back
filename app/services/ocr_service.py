"""
Google Vision API integration for OCR processing of nutrition labels.
"""

import io
import base64
from typing import Dict, Any, List, Optional, Tuple
from google.cloud import vision
from google.cloud.vision_v1 import types
import structlog

from app.core.config import settings
from app.cache.cache_manager import cache_manager

logger = structlog.get_logger(__name__)


class OCRService:
    """Google Vision API service for nutrition label text extraction."""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Vision client."""
        try:
            if settings.google_application_credentials:
                self.client = vision.ImageAnnotatorClient()
                logger.info("Google Vision client initialized successfully")
            else:
                logger.warning("Google Vision credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision client: {e}")
            self.client = None
    
    async def extract_text_from_image(
        self,
        image_data: bytes,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from nutrition label image using Google Vision API.
        
        Args:
            image_data: Raw image bytes
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.client:
            raise Exception("Google Vision client not initialized")
        
        # Check cache first
        if use_cache:
            cached_result = await cache_manager.get_ocr_result(image_data)
            if cached_result:
                logger.info("OCR result retrieved from cache")
                return cached_result
        
        try:
            # Prepare image for Vision API
            image = vision.Image(content=image_data)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")
            
            # Extract text annotations
            texts = response.text_annotations
            
            if not texts:
                return {
                    "raw_text": "",
                    "structured_data": {},
                    "confidence_scores": {},
                    "processing_metadata": {
                        "service": "google_vision",
                        "status": "no_text_found"
                    }
                }
            
            # Primary text (full text)
            full_text = texts[0].description if texts else ""
            
            # Individual text elements with bounding boxes
            text_elements = []
            for text in texts[1:]:  # Skip the first one (full text)
                vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                text_elements.append({
                    "text": text.description,
                    "bounding_box": vertices,
                    "confidence": getattr(text, 'confidence', 0.0)
                })
            
            # Structure the result
            result = {
                "raw_text": full_text,
                "text_elements": text_elements,
                "structured_data": self._structure_nutrition_data(full_text, text_elements),
                "confidence_scores": self._calculate_confidence_scores(text_elements),
                "processing_metadata": {
                    "service": "google_vision",
                    "status": "success",
                    "elements_count": len(text_elements)
                }
            }
            
            # Cache the result
            if use_cache:
                await cache_manager.set_ocr_result(image_data, result)
            
            logger.info(f"OCR processing completed: {len(text_elements)} text elements extracted")
            return result
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {
                "raw_text": "",
                "structured_data": {},
                "confidence_scores": {},
                "processing_metadata": {
                    "service": "google_vision",
                    "status": "error",
                    "error": str(e)
                }
            }
    
    def _structure_nutrition_data(
        self,
        full_text: str,
        text_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Structure extracted text into nutrition data format.
        Uses pattern matching to identify nutrition facts.
        """
        structured = {
            "product_name": None,
            "brand": None,
            "serving_size": None,
            "nutrition_facts": {},
            "ingredients": [],
            "allergens": []
        }
        
        # Convert to lowercase for pattern matching
        text_lower = full_text.lower()
        lines = full_text.split('\n')
        
        # Extract product name (usually first few lines)
        potential_names = [line.strip() for line in lines[:3] if line.strip()]
        if potential_names:
            structured["product_name"] = potential_names[0]
        
        # Extract serving size
        serving_patterns = [
            r'serving size[:\s]+([^\n]+)',
            r'per serving[:\s]+([^\n]+)',
            r'portion[:\s]+([^\n]+)'
        ]
        
        import re
        for pattern in serving_patterns:
            match = re.search(pattern, text_lower)
            if match:
                structured["serving_size"] = match.group(1).strip()
                break
        
        # Extract nutrition facts
        nutrition_patterns = {
            "calories": [r'calories[:\s]+(\d+)', r'energy[:\s]+(\d+)'],
            "protein": [r'protein[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "carbohydrates": [r'carbohydrate[s]?[:\s]+(\d+(?:\.\d+)?)\s*g', r'carbs[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "total_fat": [r'total fat[:\s]+(\d+(?:\.\d+)?)\s*g', r'fat[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "saturated_fat": [r'saturated fat[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "trans_fat": [r'trans fat[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "fiber": [r'fiber[:\s]+(\d+(?:\.\d+)?)\s*g', r'fibre[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "sugar": [r'sugar[s]?[:\s]+(\d+(?:\.\d+)?)\s*g'],
            "sodium": [r'sodium[:\s]+(\d+(?:\.\d+)?)\s*mg'],
            "potassium": [r'potassium[:\s]+(\d+(?:\.\d+)?)\s*mg'],
            "cholesterol": [r'cholesterol[:\s]+(\d+(?:\.\d+)?)\s*mg']
        }
        
        for nutrient, patterns in nutrition_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        structured["nutrition_facts"][nutrient] = value
                        break
                    except ValueError:
                        continue
        
        # Extract ingredients
        ingredients_match = re.search(r'ingredients[:\s]+([^.]+)', text_lower)
        if ingredients_match:
            ingredients_text = ingredients_match.group(1)
            # Split by common separators
            ingredients = [ing.strip() for ing in re.split(r'[,;]', ingredients_text) if ing.strip()]
            structured["ingredients"] = ingredients[:20]  # Limit to first 20 ingredients
        
        # Extract allergens
        allergen_patterns = [
            r'contains[:\s]+([^.]+)',
            r'allergens[:\s]+([^.]+)',
            r'may contain[:\s]+([^.]+)'
        ]
        
        for pattern in allergen_patterns:
            match = re.search(pattern, text_lower)
            if match:
                allergens_text = match.group(1)
                allergens = [all.strip() for all in re.split(r'[,;]', allergens_text) if all.strip()]
                structured["allergens"].extend(allergens)
        
        return structured
    
    def _calculate_confidence_scores(self, text_elements: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects of OCR."""
        if not text_elements:
            return {"overall": 0.0}
        
        # Calculate average confidence
        confidences = [elem.get("confidence", 0.0) for elem in text_elements]
        valid_confidences = [c for c in confidences if c > 0]
        
        if not valid_confidences:
            return {"overall": 0.5}  # Default confidence
        
        overall_confidence = sum(valid_confidences) / len(valid_confidences)
        
        return {
            "overall": round(overall_confidence, 3),
            "text_detection": round(overall_confidence, 3),
            "structure_extraction": round(min(overall_confidence * 0.8, 1.0), 3)  # Slightly lower for structure
        }
    
    async def validate_nutrition_label(self, image_data: bytes) -> Tuple[bool, str]:
        """
        Validate if the image contains a nutrition label.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            result = await self.extract_text_from_image(image_data, use_cache=False)
            
            if result["processing_metadata"]["status"] != "success":
                return False, "Failed to extract text from image"
            
            text_lower = result["raw_text"].lower()
            
            # Check for nutrition label indicators
            nutrition_indicators = [
                "nutrition facts", "nutrition information", "nutritional information",
                "calories", "protein", "carbohydrate", "fat", "sodium"
            ]
            
            found_indicators = sum(1 for indicator in nutrition_indicators if indicator in text_lower)
            
            if found_indicators >= 3:
                return True, "Valid nutrition label detected"
            elif found_indicators >= 1:
                return True, "Partial nutrition information detected"
            else:
                return False, "No nutrition information detected in image"
                
        except Exception as e:
            logger.error(f"Nutrition label validation failed: {e}")
            return False, f"Validation error: {str(e)}"


# Global OCR service instance
ocr_service = OCRService()
