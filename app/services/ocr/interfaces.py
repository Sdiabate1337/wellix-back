"""
OCR Service Interface and Implementation for Workflow System.
Aligned with our DI Container and Strategy Pattern architecture.

Architecture Pattern: Strategy + Dependency Injection + Interface Segregation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class OCRProvider(str, Enum):
    """Supported OCR providers."""
    GOOGLE_VISION = "google_vision"
    TESSERACT = "tesseract"
    AZURE_VISION = "azure_vision"
    AWS_TEXTRACT = "aws_textract"


@dataclass
class OCRResult:
    """Structured OCR result with confidence and metadata."""
    
    raw_text: str
    confidence: float
    detected_language: Optional[str] = None
    text_blocks: List[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    provider: OCRProvider = OCRProvider.GOOGLE_VISION
    
    # Nutrition-specific extracted data
    nutrition_keywords: List[str] = None
    detected_numbers: List[Dict[str, Any]] = None
    ingredient_sections: List[str] = None
    
    def __post_init__(self):
        if self.text_blocks is None:
            self.text_blocks = []
        if self.nutrition_keywords is None:
            self.nutrition_keywords = []
        if self.detected_numbers is None:
            self.detected_numbers = []
        if self.ingredient_sections is None:
            self.ingredient_sections = []


class IOCRService(ABC):
    """Interface for OCR services following Strategy Pattern."""
    
    @abstractmethod
    async def extract_text(self, image_data: bytes, **kwargs) -> OCRResult:
        """
        Extract text from image with nutrition-specific optimizations.
        
        Args:
            image_data: Raw image bytes
            **kwargs: Provider-specific parameters
            
        Returns:
            OCRResult with structured text and metadata
        """
        pass
    
    @abstractmethod
    async def validate_image_quality(self, image_data: bytes) -> Dict[str, Any]:
        """
        Validate image quality for OCR processing.
        
        Returns:
            Quality assessment with recommendations
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the OCR provider."""
        pass


class IOCRProcessor(ABC):
    """Interface for post-processing OCR results."""
    
    @abstractmethod
    async def enhance_nutrition_extraction(self, ocr_result: OCRResult) -> OCRResult:
        """Enhance OCR result with nutrition-specific processing."""
        pass
    
    @abstractmethod
    async def detect_nutrition_keywords(self, text: str) -> List[str]:
        """Detect nutrition-related keywords in text."""
        pass
    
    @abstractmethod
    async def extract_numerical_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical values with context (calories, grams, etc.)."""
        pass