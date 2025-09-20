"""
Google Vision OCR Service Implementation.
Production-ready service for nutrition label text extraction.

Architecture Pattern: Strategy + Template Method + Error Handling
"""

import io
import time
import asyncio
from typing import Dict, Any, List, Optional
import structlog

try:
    from google.cloud import vision
    from google.cloud.vision_v1 import types
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    vision = None
    types = None

from app.core.config import settings
from app.services.ocr.interfaces import IOCRService, OCRResult, OCRProvider


logger = structlog.get_logger(__name__)


class GoogleVisionOCRService(IOCRService):
    """
    Google Vision API implementation for OCR processing.
    
    Features:
    - High-accuracy text detection
    - Nutrition label optimization
    - Language detection
    - Confidence scoring
    - Quality validation
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.client = None
        self.credentials_path = credentials_path or getattr(settings, 'google_application_credentials', None)
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Google Vision client with error handling."""
        if not GOOGLE_VISION_AVAILABLE:
            logger.error("Google Vision API not available - install google-cloud-vision")
            return
        
        try:
            if self.credentials_path:
                # Client will use GOOGLE_APPLICATION_CREDENTIALS env var
                self.client = vision.ImageAnnotatorClient()
                logger.info("Google Vision client initialized successfully")
            else:
                logger.warning("Google Vision credentials not configured")
        except Exception as e:
            logger.error("Failed to initialize Google Vision client", error=str(e))
            self.client = None
    
    async def extract_text(self, image_data: bytes, **kwargs) -> OCRResult:
        """
        Extract text using Google Vision API with nutrition optimizations.
        
        Args:
            image_data: Raw image bytes
            **kwargs: Additional parameters (language_hints, etc.)
        """
        if not self.client:
            raise RuntimeError("Google Vision client not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare image for Vision API
            image = vision.Image(content=image_data)
            
            # Configure text detection features
            features = [
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            ]
            
            # Image context for better results
            image_context = vision.ImageContext(
                language_hints=kwargs.get('language_hints', ['en', 'fr', 'es'])
            )
            
            # Make API request
            request = vision.AnnotateImageRequest(
                image=image,
                features=features,
                image_context=image_context
            )
            
            # Execute in thread pool to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.annotate_image, request
            )
            
            if response.error.message:
                raise RuntimeError(f"Google Vision API error: {response.error.message}")
            
            # Process response
            processing_time = (time.time() - start_time) * 1000
            
            return await self._process_vision_response(response, processing_time)
            
        except Exception as e:
            logger.error("OCR extraction failed", error=str(e))
            # Return empty result with error info
            return OCRResult(
                raw_text="",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                provider=OCRProvider.GOOGLE_VISION
            )
    
    async def _process_vision_response(self, response, processing_time: float) -> OCRResult:
        """Process Google Vision API response into structured result."""
        text_annotations = response.text_annotations
        full_text_annotation = response.full_text_annotation
        
        if not text_annotations:
            return OCRResult(
                raw_text="",
                confidence=0.0,
                processing_time_ms=processing_time,
                provider=OCRProvider.GOOGLE_VISION
            )
        
        # Extract main text
        raw_text = text_annotations[0].description if text_annotations else ""
        
        # Calculate average confidence
        confidence = 0.0
        if full_text_annotation.pages:
            confidences = []
            for page in full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            confidences.append(word.confidence)
            
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Detect language
        detected_language = None
        if full_text_annotation.pages:
            for page in full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            if hasattr(word.property, 'detected_languages') and word.property.detected_languages:
                                detected_language = word.property.detected_languages[0].language_code
                                break
        
        # Extract text blocks with positions
        text_blocks = []
        for annotation in text_annotations[1:]:  # Skip first (full text)
            vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
            text_blocks.append({
                "text": annotation.description,
                "bounding_box": vertices,
                "confidence": getattr(annotation, 'confidence', 0.0)
            })
        
        # Create structured result
        result = OCRResult(
            raw_text=raw_text,
            confidence=confidence,
            detected_language=detected_language,
            text_blocks=text_blocks,
            processing_time_ms=processing_time,
            provider=OCRProvider.GOOGLE_VISION
        )
        
        # Enhance with nutrition-specific processing
        return await self._enhance_nutrition_data(result)
    
    async def _enhance_nutrition_data(self, result: OCRResult) -> OCRResult:
        """Enhance OCR result with nutrition-specific processing."""
        text = result.raw_text.lower()
        
        # Detect nutrition keywords
        nutrition_keywords = [
            "calories", "protein", "fat", "carbohydrates", "fiber", "sugar",
            "sodium", "vitamin", "mineral", "ingredients", "nutrition facts",
            "serving size", "kcal", "energie", "protéines", "glucides", "lipides"
        ]
        
        detected_keywords = [kw for kw in nutrition_keywords if kw in text]
        result.nutrition_keywords = detected_keywords
        
        # Extract numerical values with context
        import re
        number_patterns = [
            r'(\d+(?:\.\d+)?)\s*(g|mg|kcal|cal|%|ml|l)\b',
            r'(\d+(?:\.\d+)?)\s*(grams?|milligrams?|calories?|percent)\b'
        ]
        
        detected_numbers = []
        for pattern in number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected_numbers.append({
                    "value": float(match.group(1)),
                    "unit": match.group(2),
                    "context": text[max(0, match.start()-20):match.end()+20].strip()
                })
        
        result.detected_numbers = detected_numbers
        
        # Extract ingredient sections
        ingredient_markers = ["ingredients:", "ingrédients:", "composition:"]
        for marker in ingredient_markers:
            marker_pos = text.find(marker)
            if marker_pos != -1:
                # Extract text after ingredient marker (up to next section or end)
                remaining_text = result.raw_text[marker_pos + len(marker):].strip()
                lines = remaining_text.split('\n')
                ingredient_text = lines[0] if lines else ""
                
                if len(ingredient_text) > 10:  # Only if substantial text
                    result.ingredient_sections.append(ingredient_text.strip())
        
        return result
    
    async def validate_image_quality(self, image_data: bytes) -> Dict[str, Any]:
        """
        Validate image quality for OCR processing.
        
        Checks:
        - Image size and resolution
        - Blur detection (if possible)
        - Contrast estimation
        - Text region detection
        """
        try:
            # Basic size validation
            image_size = len(image_data)
            
            quality_assessment = {
                "is_valid": True,
                "warnings": [],
                "recommendations": [],
                "estimated_success_rate": 0.8  # Default
            }
            
            # Size checks
            if image_size < 10_000:  # Less than 10KB
                quality_assessment["warnings"].append("Image file size very small")
                quality_assessment["recommendations"].append("Use higher resolution image")
                quality_assessment["estimated_success_rate"] *= 0.7
            
            elif image_size > 20_000_000:  # More than 20MB
                quality_assessment["warnings"].append("Image file size very large")
                quality_assessment["recommendations"].append("Consider compressing image")
            
            # If Google Vision is available, do a quick processing test
            if self.client:
                try:
                    # Quick text detection to assess quality
                    image = vision.Image(content=image_data)
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.client.text_detection(image=image)
                    )
                    
                    if response.text_annotations:
                        confidence = len(response.text_annotations) / 50.0  # Rough estimate
                        quality_assessment["estimated_success_rate"] = min(confidence, 1.0)
                    else:
                        quality_assessment["warnings"].append("No text detected in preview")
                        quality_assessment["estimated_success_rate"] = 0.3
                        
                except Exception as e:
                    logger.debug("Quality validation failed", error=str(e))
                    quality_assessment["warnings"].append("Could not analyze image quality")
            
            return quality_assessment
            
        except Exception as e:
            logger.error("Image quality validation failed", error=str(e))
            return {
                "is_valid": False,
                "warnings": ["Failed to validate image"],
                "recommendations": ["Try a different image"],
                "estimated_success_rate": 0.0
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about Google Vision provider."""
        return {
            "provider": "Google Vision API",
            "version": "v1",
            "features": [
                "High accuracy text detection",
                "Multiple language support", 
                "Confidence scoring",
                "Block-level text extraction",
                "Language detection"
            ],
            "supported_languages": ["en", "fr", "es", "de", "it", "pt", "ja", "ko", "zh"],
            "max_image_size": "20MB",
            "supported_formats": ["JPEG", "PNG", "GIF", "BMP", "WebP", "RAW", "ICO", "PDF", "TIFF"],
            "is_available": self.client is not None
        }