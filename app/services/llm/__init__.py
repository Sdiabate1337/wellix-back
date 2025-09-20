"""
Service LLM Enrichment Package.

Ce package implémente l'architecture Multi-LLM pour l'enrichissement
des données nutritionnelles avec validation croisée et optimisation des coûts.

Architecture:
- Multi-Provider Support (GPT-4o, Claude 3.5, Gemini Pro, Llama)
- Task-Specific Routing (nutrition, allergens, health impact)
- Consensus Validation entre LLMs
- Semantic Caching pour optimisation coûts
- Knowledge Base Integration (ANSES, USDA, EFSA)
- Health Claims Validation automatique

Design Patterns:
- Interface Segregation Principle (ISP)
- Factory Pattern pour création LLM services
- Strategy Pattern pour providers multiples
- Singleton Manager pour orchestration
- Chain of Responsibility pour validation
"""

from .interfaces import (
    # Core LLM Interfaces
    ILLMService,
    ILLMEnricher,
    ITaskRouter,
    IConsensusValidator,
    
    # Caching & Knowledge
    ISemanticCache,
    IKnowledgeBase,
    IHealthClaimsValidator,
    
    # Data Classes
    LLMProvider,
    LLMTask,
    LLMResult,
    EnrichmentConfig,
    QualityMetrics,
    
    # Exceptions
    LLMServiceException,
    LLMValidationException,
    LLMQuotaException
)

from .manager import (
    LLMServiceFactory,
    LLMServiceManager,
    get_llm_manager,
    llm_manager
)

# Convenience functions pour API publique
async def enrich_product_data(product_data: dict, config: EnrichmentConfig = None) -> LLMResult:
    """
    Point d'entrée principal pour enrichissement LLM.
    
    Args:
        product_data: Données produit à enrichir
        config: Configuration optionnelle
    
    Returns:
        LLMResult: Résultat enrichi avec score qualité
    """
    manager = await get_llm_manager()
    return await manager.enrich_product_data(product_data, config)


async def analyze_nutrition_claims(text: str, validate_health_claims: bool = True) -> LLMResult:
    """
    Analyse spécifique pour claims nutritionnels.
    
    Args:
        text: Texte à analyser
        validate_health_claims: Valider les affirmations santé
    
    Returns:
        LLMResult: Analyse avec validation compliance
    """
    manager = await get_llm_manager()
    return await manager.analyze_nutrition_claims(text, validate_health_claims)


async def batch_enrich_products(products_data: list, config: EnrichmentConfig = None) -> list[LLMResult]:
    """
    Enrichissement en lot optimisé.
    
    Args:
        products_data: Liste des produits à enrichir
        config: Configuration optionnelle
    
    Returns:
        list[LLMResult]: Résultats enrichis
    """
    manager = await get_llm_manager()
    return await manager.batch_enrich_products(products_data, config)


__all__ = [
    # Core Interfaces
    "ILLMService",
    "ILLMEnricher", 
    "ITaskRouter",
    "IConsensusValidator",
    "ISemanticCache",
    "IKnowledgeBase",
    "IHealthClaimsValidator",
    
    # Data Classes
    "LLMProvider",
    "LLMTask",
    "LLMResult",
    "EnrichmentConfig",
    "QualityMetrics",
    
    # Exceptions
    "LLMServiceException",
    "LLMValidationException",
    "LLMQuotaException",
    
    # Factory & Manager
    "LLMServiceFactory",
    "LLMServiceManager",
    "get_llm_manager",
    "llm_manager",
    
    # Public API
    "enrich_product_data",
    "analyze_nutrition_claims",
    "batch_enrich_products"
]