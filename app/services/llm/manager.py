"""
Factory et Manager pour les services LLM Enrichment.
Implémente le même design pattern que les services OCR/Barcode.

Architecture Pattern : Factory + Singleton Manager + Service Locator
Inspiration : Spring IoC, Google Guice, Service Registry Pattern
"""

import asyncio
from typing import Dict, Optional, Type, Any, List
from enum import Enum
import structlog

from .interfaces import (
    ILLMService, ILLMEnricher, ITaskRouter, IConsensusValidator,
    ISemanticCache, IKnowledgeBase, IHealthClaimsValidator,
    LLMProvider, LLMTask, LLMResult, EnrichmentConfig,
    LLMServiceException, LLMValidationException
)
from .task_router import IntelligentTaskRouter
from .consensus_validator import ConsensusValidator
from .semantic_cache import SemanticCache
from .gpt4_service import GPT4Service

logger = structlog.get_logger(__name__)


class LLMServiceFactory:
    """
    Factory pour création des services LLM.
    
    Design Pattern : Factory Method + Abstract Factory
    Responsabilité : Création et configuration des services LLM selon le provider
    """
    
    # Registry des services disponibles
    _llm_services: Dict[LLMProvider, Type[ILLMService]] = {
        LLMProvider.GPT_4O: GPT4Service,
        LLMProvider.GPT_4O_MINI: GPT4Service,
        LLMProvider.GPT_4_TURBO: GPT4Service,
        # Les autres providers seront ajoutés au fur et à mesure
        # LLMProvider.CLAUDE_3_5_SONNET: Claude35Service,
        # LLMProvider.GEMINI_PRO: GeminiProService,
    }
    
    @classmethod
    def create_llm_service(cls, provider: LLMProvider, **kwargs) -> ILLMService:
        """
        Crée un service LLM selon le provider.
        
        Args:
            provider: Type de provider LLM
            **kwargs: Arguments de configuration spécifiques
            
        Returns:
            ILLMService: Instance du service LLM
        """
        service_class = cls._llm_services.get(provider)
        
        if not service_class:
            # Fallback vers GPT-4 si provider non supporté
            logger.warning(
                "Provider not supported, falling back to GPT-4O",
                requested_provider=provider,
                available_providers=list(cls._llm_services.keys())
            )
            service_class = GPT4Service
            provider = LLMProvider.GPT_4O
        
        try:
            return service_class(provider=provider, **kwargs)
        except Exception as e:
            logger.error(
                "Failed to create LLM service",
                provider=provider,
                error=str(e)
            )
            raise LLMServiceException(
                f"Failed to create {provider} service: {str(e)}",
                provider=str(provider)
            )
    
    @classmethod
    def create_task_router(cls, config: EnrichmentConfig) -> ITaskRouter:
        """
        Crée un routeur de tâches intelligent.
        
        Args:
            config: Configuration d'enrichissement
            
        Returns:
            ITaskRouter: Instance du routeur
        """
        try:
            return IntelligentTaskRouter(config)
        except Exception as e:
            logger.error("Failed to create task router", error=str(e))
            raise LLMServiceException(f"Failed to create task router: {str(e)}")
    
    @classmethod
    def create_consensus_validator(cls, min_providers: int = 2, 
                                 threshold: float = 0.8) -> IConsensusValidator:
        """
        Crée un validateur de consensus.
        
        Args:
            min_providers: Nombre minimum de providers pour validation
            threshold: Seuil de consensus
            
        Returns:
            IConsensusValidator: Instance du validateur
        """
        try:
            return ConsensusValidator(min_providers, threshold)
        except Exception as e:
            logger.error("Failed to create consensus validator", error=str(e))
            raise LLMServiceException(f"Failed to create consensus validator: {str(e)}")
    
    @classmethod
    def create_semantic_cache(cls, cache_dir: Optional[str] = None,
                            max_entries: int = 10000,
                            similarity_threshold: float = 0.95) -> ISemanticCache:
        """
        Crée un cache sémantique.
        
        Args:
            cache_dir: Répertoire de cache
            max_entries: Nombre maximum d'entrées
            similarity_threshold: Seuil de similarité
            
        Returns:
            ISemanticCache: Instance du cache
        """
        try:
            return SemanticCache(
                cache_dir=cache_dir,
                max_entries=max_entries,
                similarity_threshold=similarity_threshold
            )
        except Exception as e:
            logger.error("Failed to create semantic cache", error=str(e))
            raise LLMServiceException(f"Failed to create semantic cache: {str(e)}")
    
    @classmethod
    def register_llm_service(cls, provider: LLMProvider, service_class: Type[ILLMService]):
        """
        Enregistre un nouveau service LLM.
        
        Args:
            provider: Provider LLM
            service_class: Classe du service
        """
        cls._llm_services[provider] = service_class
        logger.info(
            "LLM service registered",
            provider=provider,
            service_class=service_class.__name__
        )


class LLMServiceManager:
    """
    Manager singleton pour orchestration des services LLM.
    
    Design Pattern : Singleton + Service Locator + Façade
    Responsabilité : Orchestration complète de l'enrichissement LLM multi-provider
    """
    
    def __init__(self, config: EnrichmentConfig):
        self.config = config
        
        # Services core
        self.task_router = LLMServiceFactory.create_task_router(config)
        self.consensus_validator = LLMServiceFactory.create_consensus_validator(
            threshold=config.consensus_threshold
        )
        self.semantic_cache = LLMServiceFactory.create_semantic_cache() if config.enable_semantic_cache else None
        
        # Registry des services LLM actifs
        self.llm_services: Dict[LLMProvider, ILLMService] = {}
        
        # Statistiques globales
        self.total_requests = 0
        self.cache_hits = 0
        self.consensus_validations = 0
        
        logger.info(
            "LLM Service Manager initialized",
            primary_provider=config.primary_provider,
            fallback_count=len(config.fallback_providers),
            cache_enabled=config.enable_semantic_cache,
            consensus_enabled=config.enable_consensus_validation
        )
    
    async def enrich_product_data(self, product_data: Dict[str, Any], 
                                config: Optional[EnrichmentConfig] = None) -> LLMResult:
        """
        Point d'entrée principal pour enrichissement LLM.
        
        Workflow complet:
        1. Vérification cache sémantique
        2. Planification des tâches d'analyse
        3. Routage et exécution parallèle
        4. Validation croisée si activée
        5. Mise en cache du résultat
        
        Args:
            product_data: Données produit à enrichir
            config: Configuration optionnelle (utilise self.config si None)
            
        Returns:
            LLMResult: Résultat enrichi avec métriques qualité
        """
        self.total_requests += 1
        effective_config = config or self.config
        
        try:
            # 1. Vérification cache sémantique
            if self.semantic_cache and effective_config.enable_semantic_cache:
                cached_result = await self.semantic_cache.get_similar_analysis(
                    product_data, effective_config.cache_similarity_threshold
                )
                
                if cached_result:
                    self.cache_hits += 1
                    logger.info(
                        "Cache hit for product analysis",
                        cache_similarity=cached_result.cache_similarity,
                        total_requests=self.total_requests,
                        cache_hit_rate=self.cache_hits / self.total_requests
                    )
                    return cached_result
            
            # 2. Planification des tâches
            planned_tasks = await self.task_router.plan_analysis(product_data)
            
            if not planned_tasks:
                raise LLMValidationException("No analysis tasks could be planned for this product")
            
            # 3. Optimisation des tâches
            if effective_config.enable_cost_optimization:
                planned_tasks = await self.task_router.optimize_for_cost(planned_tasks)
            
            # 4. Exécution des tâches
            if effective_config.parallel_tasks:
                results = await self._execute_tasks_parallel(planned_tasks, effective_config)
            else:
                results = await self._execute_tasks_sequential(planned_tasks, effective_config)
            
            # 5. Agrégation des résultats
            final_result = await self._aggregate_task_results(results)
            
            # 6. Validation croisée si requise
            if effective_config.enable_consensus_validation and final_result.task_type in [
                "nutrition_analysis", "health_impact_assessment", "claims_validation"
            ]:
                validated_result = await self._validate_with_consensus(final_result, planned_tasks[0])
                self.consensus_validations += 1
                final_result = validated_result
            
            # 7. Mise en cache
            if self.semantic_cache and effective_config.enable_semantic_cache:
                await self.semantic_cache.store_analysis(product_data, final_result)
            
            logger.info(
                "Product enrichment completed",
                tasks_executed=len(planned_tasks),
                final_quality_score=final_result.quality_metrics.overall_score,
                consensus_validated=final_result.is_validated
            )
            
            return final_result
            
        except Exception as e:
            logger.error(
                "Product enrichment failed",
                error=str(e),
                product_keys=list(product_data.keys())
            )
            raise LLMServiceException(f"Enrichment failed: {str(e)}")
    
    async def analyze_nutrition_claims(self, text: str, 
                                     validate_health_claims: bool = True) -> LLMResult:
        """
        Analyse spécifique pour claims nutritionnels.
        
        Args:
            text: Texte à analyser
            validate_health_claims: Valider les affirmations santé
            
        Returns:
            LLMResult: Analyse avec validation compliance
        """
        from .interfaces import LLMTaskType
        
        # Crée tâche spécialisée
        task = LLMTask(
            task_type=LLMTaskType.CLAIMS_VALIDATION,
            prompt=f"Analysez les affirmations nutritionnelles dans ce texte:\n\n{text}",
            data={"text": text, "validate_health_claims": validate_health_claims},
            requires_validation=validate_health_claims,
            priority=1  # Haute priorité pour compliance
        )
        
        # Exécute avec validation renforcée
        result = await self._execute_single_task(task, self.config)
        
        if validate_health_claims and self.config.enable_health_claims_validation:
            # Validation additionnelle des health claims
            # (sera implémentée avec IHealthClaimsValidator)
            pass
        
        return result
    
    async def batch_enrich_products(self, products_data: List[Dict[str, Any]], 
                                  config: Optional[EnrichmentConfig] = None) -> List[LLMResult]:
        """
        Enrichissement en lot optimisé.
        
        Args:
            products_data: Liste des produits à enrichir
            config: Configuration optionnelle
            
        Returns:
            List[LLMResult]: Résultats enrichis
        """
        effective_config = config or self.config
        
        # Traitement par batches pour éviter surcharge
        batch_size = min(effective_config.max_concurrent_requests, len(products_data))
        results = []
        
        for i in range(0, len(products_data), batch_size):
            batch = products_data[i:i + batch_size]
            
            batch_tasks = [
                self.enrich_product_data(product_data, effective_config)
                for product_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Gère les exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Batch enrichment failed for product",
                        batch_index=i + j,
                        error=str(result)
                    )
                    # Crée résultat d'erreur
                    error_result = self._create_error_result(str(result))
                    results.append(error_result)
                else:
                    results.append(result)
        
        logger.info(
            "Batch enrichment completed",
            total_products=len(products_data),
            successful=len([r for r in results if not hasattr(r, 'error')]),
            failed=len([r for r in results if hasattr(r, 'error')])
        )
        
        return results
    
    async def _get_llm_service(self, provider: LLMProvider) -> ILLMService:
        """Récupère ou crée un service LLM."""
        if provider not in self.llm_services:
            try:
                service = LLMServiceFactory.create_llm_service(provider)
                self.llm_services[provider] = service
                logger.debug(f"Created LLM service for {provider}")
            except Exception as e:
                logger.error(f"Failed to create service for {provider}: {e}")
                raise
        
        return self.llm_services[provider]
    
    async def _execute_tasks_parallel(self, tasks: List[LLMTask], 
                                    config: EnrichmentConfig) -> List[LLMResult]:
        """Exécute les tâches en parallèle."""
        semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        async def execute_with_semaphore(task):
            async with semaphore:
                return await self._execute_single_task(task, config)
        
        return await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
    
    async def _execute_tasks_sequential(self, tasks: List[LLMTask], 
                                      config: EnrichmentConfig) -> List[LLMResult]:
        """Exécute les tâches en séquentiel."""
        results = []
        for task in tasks:
            try:
                result = await self._execute_single_task(task, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                error_result = self._create_error_result(str(e))
                results.append(error_result)
        
        return results
    
    async def _execute_single_task(self, task: LLMTask, config: EnrichmentConfig) -> LLMResult:
        """Exécute une tâche LLM unique."""
        # Route vers le provider optimal
        optimal_provider = await self.task_router.route_task(task)
        
        # Récupère le service
        llm_service = await self._get_llm_service(optimal_provider)
        
        # Exécute la tâche
        result = await llm_service.analyze(task)
        
        # Enregistre le résultat pour améliorer le routage
        await self.task_router.record_task_result(
            optimal_provider,
            result.processing_time,
            result.quality_metrics.overall_score,
            True
        )
        
        return result
    
    async def _aggregate_task_results(self, results: List[LLMResult]) -> LLMResult:
        """Agrège les résultats de multiples tâches."""
        # Filtre les erreurs
        valid_results = [r for r in results if isinstance(r, LLMResult) and not hasattr(r, 'error')]
        
        if not valid_results:
            raise LLMValidationException("No valid results to aggregate")
        
        # Utilise le résultat principal (nutrition_analysis en priorité)
        primary_result = None
        for result in valid_results:
            if "nutrition_analysis" in str(result.task_type):
                primary_result = result
                break
        
        if not primary_result:
            primary_result = valid_results[0]
        
        # Enrichit avec données des autres tâches
        aggregated_analysis = primary_result.analysis.copy() if isinstance(primary_result.analysis, dict) else {}
        
        for result in valid_results:
            if result != primary_result and isinstance(result.analysis, dict):
                # Merge les données non conflictuelles
                for key, value in result.analysis.items():
                    if key not in aggregated_analysis:
                        aggregated_analysis[key] = value
        
        # Crée le résultat agrégé
        from .interfaces import LLMResult, LLMProvider
        
        aggregated_result = LLMResult(
            analysis=aggregated_analysis,
            raw_response=primary_result.raw_response,
            quality_metrics=primary_result.quality_metrics,
            confidence_score=primary_result.confidence_score,
            provider_used=LLMProvider.FALLBACK,  # Indique agrégation
            task_type=primary_result.task_type,
            processing_time=sum(r.processing_time for r in valid_results) / len(valid_results)
        )
        
        return aggregated_result
    
    async def _validate_with_consensus(self, result: LLMResult, original_task: LLMTask) -> LLMResult:
        """Valide un résultat avec consensus."""
        try:
            validated_result = await self.consensus_validator.validate_result(result, original_task)
            return validated_result
        except Exception as e:
            logger.warning(f"Consensus validation failed: {e}")
            # Retourne résultat original si validation échoue
            result.is_validated = False
            return result
    
    def _create_error_result(self, error_message: str) -> LLMResult:
        """Crée un résultat d'erreur."""
        from .interfaces import LLMResult, LLMProvider, LLMTaskType, QualityMetrics
        
        return LLMResult(
            analysis={"error": error_message, "confidence": 0.0},
            raw_response=f"Error: {error_message}",
            quality_metrics=QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
            confidence_score=0.0,
            provider_used=LLMProvider.FALLBACK,
            task_type=LLMTaskType.NUTRITION_ANALYSIS,
            processing_time=0.0
        )
    
    async def cleanup(self):
        """Nettoie les ressources."""
        # Ferme tous les services LLM
        for service in self.llm_services.values():
            if hasattr(service, 'close'):
                await service.close()
        
        # Nettoie le cache
        if self.semantic_cache and hasattr(self.semantic_cache, 'close'):
            await self.semantic_cache.close()
        
        logger.info("LLM Service Manager cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du manager."""
        stats = {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_requests),
            "consensus_validations": self.consensus_validations,
            "active_providers": list(self.llm_services.keys()),
            "config": {
                "primary_provider": self.config.primary_provider,
                "fallback_providers": self.config.fallback_providers,
                "cache_enabled": self.config.enable_semantic_cache,
                "consensus_enabled": self.config.enable_consensus_validation
            }
        }
        
        # Ajoute stats des services individuels
        if self.semantic_cache and hasattr(self.semantic_cache, 'get_cache_stats'):
            stats["cache_stats"] = self.semantic_cache.get_cache_stats()
        
        return stats


# Singleton instance
_llm_manager: Optional[LLMServiceManager] = None


async def get_llm_manager(config: Optional[EnrichmentConfig] = None) -> LLMServiceManager:
    """
    Récupère l'instance singleton du LLM Service Manager.
    
    Args:
        config: Configuration optionnelle (utilisée seulement à la première création)
        
    Returns:
        LLMServiceManager: Instance singleton
    """
    global _llm_manager
    
    if _llm_manager is None:
        if config is None:
            # Configuration par défaut
            config = EnrichmentConfig()
        
        _llm_manager = LLMServiceManager(config)
        logger.info("LLM Service Manager singleton created")
    
    return _llm_manager


# Alias pour accès direct
llm_manager = get_llm_manager