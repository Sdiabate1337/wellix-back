"""
OCR Service Package for Workflow System.
Provides production-ready OCR capabilities with nutrition optimization.

Components:
- IOCRService: Interface for OCR providers
- GoogleVisionOCRService: Google Vision API implementation
- NutritionOCRProcessor: Nutrition-specific post-processing
- OCRServiceManager: Service lifecycle management
"""

from app.services.ocr.interfaces import IOCRService, IOCRProcessor, OCRResult, OCRProvider
from app.services.ocr.google_vision import GoogleVisionOCRService
from app.services.ocr.nutrition_processor import NutritionOCRProcessor
from app.services.ocr.manager import OCRServiceFactory, OCRServiceManager, ocr_manager

__all__ = [
    # Interfaces
    "IOCRService",
    "IOCRProcessor", 
    "OCRResult",
    "OCRProvider",
    
    # Implementations
    "GoogleVisionOCRService",
    "NutritionOCRProcessor",
    
    # Management
    "OCRServiceFactory",
    "OCRServiceManager",
    "ocr_manager"
]