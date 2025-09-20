"""
Service de code-barres pour Wellix.
Lookup de produits alimentaires via OpenFoodFacts et autres providers.

Architecture : Factory Pattern + Strategy Pattern + Service Locator
Usage : Interface unifiée pour différents providers de codes-barres

Example:
    from app.services.barcode import barcode_manager, lookup_barcode
    
    # Lookup simple
    result = await lookup_barcode("3017620422003")
    
    # Recherche de produits
    results = await barcode_manager.search_with_enrichment("nutella")
    
    # Configuration personnalisée
    manager = await get_barcode_manager()
    manager.configure(language="en", country="usa")
"""

from .interfaces import (
    IBarcodeService,
    IBarcodeEnricher, 
    BarcodeResult,
    BarcodeProvider,
    BarcodeServiceError,
    BarcodeNotFoundError,
    BarcodeValidationError,
    BarcodeRateLimitError
)

from .openfoodfacts import OpenFoodFactsService
from .enricher import NutritionDataEnricher
from .manager import (
    BarcodeServiceFactory,
    BarcodeServiceManager,
    get_barcode_manager,
    lookup_barcode,
    search_products
)

# Exports publics
__all__ = [
    # Interfaces
    "IBarcodeService",
    "IBarcodeEnricher",
    "BarcodeResult", 
    "BarcodeProvider",
    
    # Exceptions
    "BarcodeServiceError",
    "BarcodeNotFoundError", 
    "BarcodeValidationError",
    "BarcodeRateLimitError",
    
    # Implémentations
    "OpenFoodFactsService",
    "NutritionDataEnricher",
    
    # Factory et Manager
    "BarcodeServiceFactory",
    "BarcodeServiceManager",
    "get_barcode_manager",
    
    # Fonctions utilitaires
    "lookup_barcode",
    "search_products"
]

# Initialisation lazy du manager global
_manager_instance = None


async def barcode_manager() -> BarcodeServiceManager:
    """
    Accès global au manager de services de code-barres.
    
    Returns:
        Instance singleton du BarcodeServiceManager
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = await get_barcode_manager()
    return _manager_instance