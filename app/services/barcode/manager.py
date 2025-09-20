"""
Factory et Manager pour les services de code-barres.
Implémente le même design pattern que les services OCR.

Architecture Pattern : Factory + Singleton Manager + Service Locator
Inspiration : Spring IoC, Google Guice, Service Registry Pattern
"""

import asyncio
from typing import Dict, Optional, Type, Any, List
from enum import Enum
import structlog

from .interfaces import IBarcodeService, IBarcodeEnricher, BarcodeProvider
from .openfoodfacts import OpenFoodFactsService
from .enricher import NutritionDataEnricher

logger = structlog.get_logger(__name__)


class BarcodeServiceFactory:
    """
    Factory pour créer les services de code-barres selon le provider choisi.
    
    Pattern : Factory Method + Strategy
    Extensible pour ajouter d'autres providers (UPC Database, etc.)
    """
    
    @staticmethod
    def create_barcode_service(provider: BarcodeProvider, **kwargs) -> IBarcodeService:
        """
        Crée un service de code-barres selon le provider.
        
        Args:
            provider: Type de provider (OPENFOODFACTS, UPC_DATABASE, etc.)
            **kwargs: Configuration spécifique au provider
            
        Returns:
            Instance du service de code-barres
            
        Raises:
            ValueError: Si le provider n'est pas supporté
        """
        if provider == BarcodeProvider.OPENFOODFACTS:
            return OpenFoodFactsService(
                language=kwargs.get("language", "fr"),
                country=kwargs.get("country", "france"),
                timeout=kwargs.get("timeout", 10),
                user_agent=kwargs.get("user_agent", "WellixApp/1.0")
            )
        
        elif provider == BarcodeProvider.UPC_DATABASE:
            # TODO: Implémenter UPCDatabaseService
            raise NotImplementedError("UPC Database service not yet implemented")
        
        elif provider == BarcodeProvider.BARCODE_LOOKUP:
            # TODO: Implémenter BarcodeLookupService  
            raise NotImplementedError("Barcode Lookup service not yet implemented")
        
        elif provider == BarcodeProvider.FALLBACK:
            # Service de fallback avec données minimales
            return _FallbackBarcodeService()
        
        else:
            raise ValueError(f"Unsupported barcode provider: {provider}")
    
    @staticmethod
    def create_barcode_enricher(enricher_type: str = "nutrition") -> IBarcodeEnricher:
        """
        Crée un enrichisseur de données selon le type.
        
        Args:
            enricher_type: Type d'enrichisseur (nutrition, allergen, etc.)
            
        Returns:
            Instance de l'enrichisseur
        """
        if enricher_type == "nutrition":
            return NutritionDataEnricher()
        else:
            raise ValueError(f"Unsupported enricher type: {enricher_type}")


class _FallbackBarcodeService(IBarcodeService):
    """Service de fallback pour codes-barres non trouvés."""
    
    @property
    def provider_name(self) -> str:
        return "Fallback"
    
    @property
    def supports_nutrition_data(self) -> bool:
        return False
    
    async def lookup_product(self, barcode: str):
        """Retourne des données minimales."""
        from .interfaces import BarcodeResult
        return BarcodeResult(
            barcode=barcode,
            product_name=f"Unknown Product ({barcode})",
            provider=self.provider_name,
            confidence=0.1,
            data_quality="poor"
        )
    
    async def validate_barcode(self, barcode: str):
        return {"is_valid": True, "format_type": "unknown", "warnings": []}
    
    async def search_products(self, query: str, limit: int = 10):
        return []


class BarcodeServiceManager:
    """
    Manager singleton pour les services de code-barres.
    
    Responsabilités :
    - Gestion du cycle de vie des services
    - Configuration centralisée
    - Cache des instances de services
    - Fallback automatique entre providers
    """
    
    _instance: Optional['BarcodeServiceManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        """Initialise le manager avec configuration par défaut."""
        if BarcodeServiceManager._instance is not None:
            raise RuntimeError("BarcodeServiceManager is a singleton")
        
        # Configuration par défaut
        self.config = {
            "primary_provider": BarcodeProvider.OPENFOODFACTS,
            "fallback_providers": [BarcodeProvider.FALLBACK],
            "enable_enrichment": True,
            "cache_enabled": True,
            "timeout": 10,
            "language": "fr",
            "country": "france"
        }
        
        # Cache des services instanciés
        self._service_cache: Dict[BarcodeProvider, IBarcodeService] = {}
        self._enricher_cache: Dict[str, IBarcodeEnricher] = {}
        
        # Service principal
        self._primary_service: Optional[IBarcodeService] = None
        self._enricher: Optional[IBarcodeEnricher] = None
        
        logger.info("Barcode service manager initialized", config=self.config)
    
    @classmethod
    async def get_instance(cls) -> 'BarcodeServiceManager':
        """Obtient l'instance singleton (thread-safe)."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def configure(self, **config_updates):
        """
        Met à jour la configuration du manager.
        
        Args:
            **config_updates: Nouvelles valeurs de configuration
        """
        self.config.update(config_updates)
        
        # Invalidation du cache si configuration change
        self._service_cache.clear()
        self._enricher_cache.clear()
        self._primary_service = None
        self._enricher = None
        
        logger.info("Barcode service manager reconfigured", new_config=self.config)
    
    def get_barcode_service(self, provider: Optional[BarcodeProvider] = None) -> IBarcodeService:
        """
        Obtient un service de code-barres.
        
        Args:
            provider: Provider spécifique ou None pour le provider principal
            
        Returns:
            Instance du service de code-barres
        """
        target_provider = provider or self.config["primary_provider"]
        
        # Cache lookup
        if target_provider in self._service_cache:
            return self._service_cache[target_provider]
        
        # Création du service
        try:
            service = BarcodeServiceFactory.create_barcode_service(
                target_provider,
                language=self.config["language"],
                country=self.config["country"],
                timeout=self.config["timeout"]
            )
            
            # Mise en cache
            self._service_cache[target_provider] = service
            
            logger.info(
                "Barcode service created",
                provider=target_provider.value,
                supports_nutrition=service.supports_nutrition_data
            )
            
            return service
            
        except Exception as e:
            logger.error(
                "Failed to create barcode service",
                provider=target_provider.value,
                error=str(e)
            )
            
            # Fallback vers service de base
            if target_provider != BarcodeProvider.FALLBACK:
                return self.get_barcode_service(BarcodeProvider.FALLBACK)
            else:
                raise
    
    def get_enricher(self, enricher_type: str = "nutrition") -> IBarcodeEnricher:
        """
        Obtient un enrichisseur de données.
        
        Args:
            enricher_type: Type d'enrichisseur
            
        Returns:
            Instance de l'enrichisseur
        """
        if enricher_type in self._enricher_cache:
            return self._enricher_cache[enricher_type]
        
        enricher = BarcodeServiceFactory.create_barcode_enricher(enricher_type)
        self._enricher_cache[enricher_type] = enricher
        
        logger.info("Barcode enricher created", type=enricher_type)
        
        return enricher
    
    async def lookup_with_fallback(self, barcode: str) -> Optional['BarcodeResult']:
        """
        Recherche avec fallback automatique entre providers.
        
        Args:
            barcode: Code-barres à rechercher
            
        Returns:
            BarcodeResult du premier provider qui trouve le produit
        """
        # Provider principal
        primary_service = self.get_barcode_service()
        
        try:
            result = await primary_service.lookup_product(barcode)
            
            if result:
                # Enrichissement si activé
                if self.config["enable_enrichment"]:
                    enricher = self.get_enricher()
                    result = await enricher.enrich_product_data(result)
                
                logger.info(
                    "Product found with primary provider",
                    barcode=barcode,
                    provider=primary_service.provider_name,
                    confidence=result.confidence
                )
                
                return result
                
        except Exception as e:
            logger.warning(
                "Primary provider failed",
                barcode=barcode,
                provider=primary_service.provider_name,
                error=str(e)
            )
        
        # Fallback providers
        for fallback_provider in self.config["fallback_providers"]:
            try:
                fallback_service = self.get_barcode_service(fallback_provider)
                result = await fallback_service.lookup_product(barcode)
                
                if result:
                    logger.info(
                        "Product found with fallback provider",
                        barcode=barcode,
                        provider=fallback_service.provider_name,
                        confidence=result.confidence
                    )
                    return result
                    
            except Exception as e:
                logger.warning(
                    "Fallback provider failed",
                    barcode=barcode,
                    provider=fallback_provider.value,
                    error=str(e)
                )
        
        logger.info("Product not found in any provider", barcode=barcode)
        return None
    
    async def search_with_enrichment(self, query: str, limit: int = 10) -> List['BarcodeResult']:
        """
        Recherche de produits avec enrichissement automatique.
        
        Args:
            query: Requête de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            Liste de BarcodeResult enrichis
        """
        service = self.get_barcode_service()
        results = await service.search_products(query, limit)
        
        # Enrichissement si activé
        if self.config["enable_enrichment"] and results:
            enricher = self.get_enricher()
            enriched_results = []
            
            for result in results:
                try:
                    enriched_result = await enricher.enrich_product_data(result)
                    enriched_results.append(enriched_result)
                except Exception as e:
                    logger.warning(
                        "Failed to enrich search result",
                        barcode=result.barcode,
                        error=str(e)
                    )
                    enriched_results.append(result)  # Résultat non enrichi
            
            return enriched_results
        
        return results
    
    async def close_all_services(self):
        """Ferme toutes les connexions des services."""
        for service in self._service_cache.values():
            if hasattr(service, 'close'):
                try:
                    await service.close()
                except Exception as e:
                    logger.warning("Failed to close service", error=str(e))
        
        self._service_cache.clear()
        self._enricher_cache.clear()
        
        logger.info("All barcode services closed")


# Instance globale singleton
barcode_manager: Optional[BarcodeServiceManager] = None


async def get_barcode_manager() -> BarcodeServiceManager:
    """
    Obtient l'instance globale du manager de services de code-barres.
    
    Returns:
        Instance singleton du BarcodeServiceManager
    """
    global barcode_manager
    if barcode_manager is None:
        barcode_manager = await BarcodeServiceManager.get_instance()
    return barcode_manager


# Fonctions utilitaires pour accès rapide
async def lookup_barcode(barcode: str):
    """Lookup rapide d'un code-barres."""
    manager = await get_barcode_manager()
    return await manager.lookup_with_fallback(barcode)


async def search_products(query: str, limit: int = 10):
    """Recherche rapide de produits."""
    manager = await get_barcode_manager()
    return await manager.search_with_enrichment(query, limit)