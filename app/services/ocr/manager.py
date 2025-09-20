"""
OCR Service Factory and DI Integration.
Provides OCR services to the workflow system.

Architecture Pattern: Factory + Dependency Injection + Configuration
"""

from typing import Dict, Any, Optional
from enum import Enum
import structlog

from app.services.ocr.interfaces import IOCRService, IOCRProcessor, OCRProvider
from app.services.ocr.google_vision import GoogleVisionOCRService
from app.services.ocr.nutrition_processor import NutritionOCRProcessor
from app.core.config import settings


logger = structlog.get_logger(__name__)


class OCRServiceFactory:
    """
    Factory for creating OCR services based on configuration.
    
    Supports multiple providers with fallback strategies.
    """
    
    @staticmethod
    def create_ocr_service(
        provider: OCRProvider = OCRProvider.GOOGLE_VISION,
        **kwargs
    ) -> IOCRService:
        """
        Create OCR service instance based on provider.
        
        Args:
            provider: OCR provider to use
            **kwargs: Provider-specific configuration
            
        Returns:
            IOCRService implementation
        """
        if provider == OCRProvider.GOOGLE_VISION:
            return GoogleVisionOCRService(
                credentials_path=kwargs.get('credentials_path')
            )
        elif provider == OCRProvider.TESSERACT:
            # Placeholder for Tesseract implementation
            logger.warning("Tesseract OCR not implemented yet")
            return GoogleVisionOCRService()  # Fallback
        else:
            logger.warning(f"Unknown OCR provider: {provider}, using Google Vision")
            return GoogleVisionOCRService()
    
    @staticmethod
    def create_nutrition_processor() -> IOCRProcessor:
        """Create nutrition-specific OCR processor."""
        return NutritionOCRProcessor()
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict[str, Any]]:
        """Get information about available OCR providers."""
        providers = {}
        
        # Check Google Vision availability
        try:
            google_service = GoogleVisionOCRService()
            providers[OCRProvider.GOOGLE_VISION] = google_service.get_provider_info()
        except Exception as e:
            providers[OCRProvider.GOOGLE_VISION] = {
                "provider": "Google Vision API",
                "is_available": False,
                "error": str(e)
            }
        
        return providers


class OCRServiceManager:
    """
    Manages OCR service lifecycle and configuration.
    
    Integrates with the workflow DI container.
    """
    
    def __init__(self):
        self._ocr_service: Optional[IOCRService] = None
        self._processor: Optional[IOCRProcessor] = None
        self._provider = self._get_configured_provider()
    
    def _get_configured_provider(self) -> OCRProvider:
        """Get OCR provider from configuration."""
        provider_name = getattr(settings, 'ocr_provider', 'google_vision').lower()
        
        provider_mapping = {
            'google_vision': OCRProvider.GOOGLE_VISION,
            'tesseract': OCRProvider.TESSERACT,
            'azure_vision': OCRProvider.AZURE_VISION,
            'aws_textract': OCRProvider.AWS_TEXTRACT
        }
        
        return provider_mapping.get(provider_name, OCRProvider.GOOGLE_VISION)
    
    def get_ocr_service(self) -> IOCRService:
        """Get or create OCR service instance (singleton)."""
        if self._ocr_service is None:
            self._ocr_service = OCRServiceFactory.create_ocr_service(
                provider=self._provider
            )
            logger.info(f"OCR service created", provider=self._provider.value)
        
        return self._ocr_service
    
    def get_nutrition_processor(self) -> IOCRProcessor:
        """Get or create nutrition processor instance (singleton)."""
        if self._processor is None:
            self._processor = OCRServiceFactory.create_nutrition_processor()
            logger.info("Nutrition OCR processor created")
        
        return self._processor
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about current OCR configuration."""
        return {
            "current_provider": self._provider.value,
            "service_available": self._ocr_service is not None,
            "processor_available": self._processor is not None,
            "available_providers": OCRServiceFactory.get_available_providers()
        }


# Global instance for DI container
ocr_manager = OCRServiceManager()