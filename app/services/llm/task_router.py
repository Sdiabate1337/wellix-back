"""
Task Router intelligent pour optimisation Multi-LLM.

Implémente le routage intelligent des tâches vers les providers LLM optimaux
selon la spécialisation, le coût et la performance.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog

from .interfaces import (
    ITaskRouter, LLMProvider, LLMTaskType, LLMTask, 
    EnrichmentConfig, LLMServiceException
)

logger = structlog.get_logger(__name__)


@dataclass
class ProviderPerformance:
    """Métriques de performance d'un provider."""
    
    provider: LLMProvider
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    avg_quality_score: float = 0.0
    cost_per_token: float = 0.0
    current_load: int = 0
    max_concurrent: int = 10
    
    # Historique des performances
    recent_requests: List[Tuple[datetime, float, float]] = field(default_factory=list)  # (timestamp, duration, quality)
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    @property
    def is_available(self) -> bool:
        """Indique si le provider est disponible."""
        return (
            self.current_load < self.max_concurrent and
            self.consecutive_failures < 3 and
            (self.last_failure is None or 
             datetime.now() - self.last_failure > timedelta(minutes=5))
        )
    
    @property
    def efficiency_score(self) -> float:
        """Score d'efficacité combiné (qualité/coût/vitesse)."""
        if not self.recent_requests:
            return 0.5  # Score neutre pour nouveaux providers
        
        # Normalisation des métriques (0-1)
        quality_norm = min(self.avg_quality_score, 1.0)
        speed_norm = max(0, min(1.0, 1.0 - (self.avg_response_time / 30.0)))  # 30s = 0 score
        reliability_norm = self.success_rate
        
        # Pénalité coût (plus cher = score plus bas)
        cost_penalty = max(0, min(1.0, 1.0 - (self.cost_per_token / 0.05)))  # 0.05$ = pénalité max
        
        # Score pondéré
        efficiency = (
            quality_norm * 0.4 +
            speed_norm * 0.25 +
            reliability_norm * 0.25 +
            cost_penalty * 0.1
        )
        
        return efficiency
    
    def record_request(self, duration: float, quality_score: float, success: bool):
        """Enregistre une requête pour mise à jour des métriques."""
        now = datetime.now()
        
        if success:
            self.recent_requests.append((now, duration, quality_score))
            self.consecutive_failures = 0
            
            # Garde seulement les 50 dernières requêtes
            if len(self.recent_requests) > 50:
                self.recent_requests = self.recent_requests[-50:]
            
            # Recalcul des moyennes
            recent_durations = [req[1] for req in self.recent_requests]
            recent_qualities = [req[2] for req in self.recent_requests]
            
            self.avg_response_time = sum(recent_durations) / len(recent_durations)
            self.avg_quality_score = sum(recent_qualities) / len(recent_qualities)
            self.success_rate = min(1.0, self.success_rate * 0.9 + 0.1)  # Smooth update
        else:
            self.last_failure = now
            self.consecutive_failures += 1
            self.success_rate = max(0.0, self.success_rate * 0.9)  # Dégrade le taux de succès


class IntelligentTaskRouter(ITaskRouter):
    """
    Routeur de tâches intelligent avec optimisation multi-critères.
    
    Fonctionnalités:
    - Routage par spécialisation du provider
    - Optimisation coût/performance
    - Load balancing automatique
    - Fallback intelligent en cas d'échec
    - Apprentissage des performances
    """
    
    # Spécialisations des providers par type de tâche
    TASK_SPECIALIZATIONS = {
        LLMTaskType.NUTRITION_ANALYSIS: [
            LLMProvider.GPT_4O,
            LLMProvider.CLAUDE_3_5_SONNET,
            LLMProvider.GEMINI_PRO
        ],
        LLMTaskType.ALLERGEN_DETECTION: [
            LLMProvider.CLAUDE_3_5_SONNET,  # Excellent pour safety
            LLMProvider.GPT_4O,
            LLMProvider.GEMINI_PRO
        ],
        LLMTaskType.HEALTH_IMPACT_ASSESSMENT: [
            LLMProvider.GPT_4O,  # Meilleur raisonnement
            LLMProvider.CLAUDE_3_5_SONNET,
            LLMProvider.GEMINI_PRO
        ],
        LLMTaskType.INGREDIENT_PARSING: [
            LLMProvider.GEMINI_PRO,  # Bon rapport qualité/prix
            LLMProvider.GPT_4O_MINI,
            LLMProvider.CLAUDE_3_5_HAIKU
        ],
        LLMTaskType.DIETARY_COMPLIANCE: [
            LLMProvider.CLAUDE_3_5_SONNET,  # Excellent pour conformité
            LLMProvider.GPT_4O,
            LLMProvider.GEMINI_PRO
        ],
        LLMTaskType.PRODUCT_CATEGORIZATION: [
            LLMProvider.GEMINI_PRO,
            LLMProvider.GPT_4O_MINI,
            LLMProvider.CLAUDE_3_5_HAIKU
        ],
        LLMTaskType.CLAIMS_VALIDATION: [
            LLMProvider.CLAUDE_3_5_SONNET,  # Best for compliance
            LLMProvider.GPT_4O,
            LLMProvider.GEMINI_PRO
        ],
        LLMTaskType.RECIPE_ANALYSIS: [
            LLMProvider.GPT_4O,
            LLMProvider.CLAUDE_3_5_SONNET,
            LLMProvider.GEMINI_PRO
        ],
        LLMTaskType.NUTRITIONAL_COMPARISON: [
            LLMProvider.GPT_4O,
            LLMProvider.CLAUDE_3_5_SONNET,
            LLMProvider.GEMINI_PRO
        ]
    }
    
    # Coûts approximatifs par token (input/output moyen en USD)
    PROVIDER_COSTS = {
        LLMProvider.GPT_4O: 0.00003,
        LLMProvider.GPT_4O_MINI: 0.00000015,
        LLMProvider.GPT_4_TURBO: 0.00002,
        LLMProvider.CLAUDE_3_5_SONNET: 0.00003,
        LLMProvider.CLAUDE_3_5_HAIKU: 0.00000025,
        LLMProvider.CLAUDE_3_OPUS: 0.000075,
        LLMProvider.GEMINI_PRO: 0.0000007,
        LLMProvider.GEMINI_FLASH: 0.00000035,
        LLMProvider.GEMINI_ULTRA: 0.00006,
        LLMProvider.LLAMA_70B: 0.0,  # Local
        LLMProvider.LLAMA_8B: 0.0,   # Local
        LLMProvider.FALLBACK: 0.00001
    }
    
    def __init__(self, config: EnrichmentConfig):
        self.config = config
        self.performance_tracker: Dict[LLMProvider, ProviderPerformance] = {}
        self.load_balancer: Dict[LLMProvider, int] = {}
        
        # Initialisation des métriques de performance
        self._initialize_performance_tracking()
        
        logger.info(
            "Task Router initialized",
            primary_provider=config.primary_provider,
            fallback_count=len(config.fallback_providers),
            cost_optimization=config.enable_cost_optimization
        )
    
    def _initialize_performance_tracking(self):
        """Initialise le tracking de performance pour tous les providers."""
        for provider in LLMProvider:
            if provider != LLMProvider.FALLBACK:
                self.performance_tracker[provider] = ProviderPerformance(
                    provider=provider,
                    cost_per_token=self.PROVIDER_COSTS.get(provider, 0.00001)
                )
                self.load_balancer[provider] = 0
    
    async def route_task(self, task: LLMTask) -> LLMProvider:
        """
        Route une tâche vers le provider optimal.
        
        Algorithme:
        1. Filtre les providers spécialisés pour la tâche
        2. Évalue disponibilité et performance
        3. Optimise selon critères (coût/qualité/vitesse)
        4. Sélectionne le provider optimal
        """
        try:
            # 1. Récupère les providers spécialisés
            specialized_providers = self._get_specialized_providers(task.task_type)
            
            # 2. Filtre les providers disponibles
            available_providers = [
                provider for provider in specialized_providers
                if self._is_provider_available(provider)
            ]
            
            if not available_providers:
                logger.warning(
                    "No specialized providers available, using fallbacks",
                    task_type=task.task_type,
                    specialized_count=len(specialized_providers)
                )
                available_providers = self._get_fallback_providers()
            
            # 3. Sélectionne le provider optimal
            optimal_provider = await self._select_optimal_provider(
                available_providers, task
            )
            
            # 4. Met à jour le load balancer
            self.load_balancer[optimal_provider] += 1
            
            logger.info(
                "Task routed successfully",
                task_type=task.task_type,
                selected_provider=optimal_provider,
                available_options=len(available_providers)
            )
            
            return optimal_provider
            
        except Exception as e:
            logger.error(
                "Task routing failed",
                task_type=task.task_type,
                error=str(e)
            )
            # Fallback vers le provider primaire
            return self.config.primary_provider
    
    async def plan_analysis(self, product_data: Dict[str, Any]) -> List[LLMTask]:
        """
        Planifie l'analyse complète d'un produit.
        
        Crée les tâches optimales selon les données disponibles.
        """
        tasks = []
        
        # Analyse nutritionnelle de base (toujours)
        if product_data.get("nutrition_facts") or product_data.get("ingredients"):
            nutrition_task = LLMTask(
                task_type=LLMTaskType.NUTRITION_ANALYSIS,
                prompt=self._build_nutrition_prompt(product_data),
                data=product_data,
                priority=1  # Haute priorité
            )
            tasks.append(nutrition_task)
        
        # Détection d'allergènes si ingrédients disponibles
        if product_data.get("ingredients"):
            allergen_task = LLMTask(
                task_type=LLMTaskType.ALLERGEN_DETECTION,
                prompt=self._build_allergen_prompt(product_data),
                data=product_data,
                priority=1
            )
            tasks.append(allergen_task)
        
        # Évaluation impact santé
        if product_data.get("nutrition_facts"):
            health_task = LLMTask(
                task_type=LLMTaskType.HEALTH_IMPACT_ASSESSMENT,
                prompt=self._build_health_impact_prompt(product_data),
                data=product_data,
                priority=2
            )
            tasks.append(health_task)
        
        # Catégorisation produit
        if product_data.get("product_name") or product_data.get("category"):
            category_task = LLMTask(
                task_type=LLMTaskType.PRODUCT_CATEGORIZATION,
                prompt=self._build_categorization_prompt(product_data),
                data=product_data,
                priority=3
            )
            tasks.append(category_task)
        
        # Validation des claims si présentes
        if product_data.get("marketing_claims") or product_data.get("health_claims"):
            claims_task = LLMTask(
                task_type=LLMTaskType.CLAIMS_VALIDATION,
                prompt=self._build_claims_validation_prompt(product_data),
                data=product_data,
                priority=1,  # Haute priorité pour compliance
                requires_validation=True
            )
            tasks.append(claims_task)
        
        logger.info(
            "Analysis plan created",
            task_count=len(tasks),
            high_priority_tasks=len([t for t in tasks if t.priority == 1])
        )
        
        return tasks
    
    async def optimize_for_cost(self, tasks: List[LLMTask]) -> List[LLMTask]:
        """
        Optimise les tâches pour minimiser les coûts.
        
        Stratégies:
        - Groupement des tâches compatibles
        - Utilisation de providers moins chers pour tâches simples
        - Priorisation des tâches critiques
        """
        if not self.config.enable_cost_optimization:
            return tasks
        
        optimized_tasks = []
        
        for task in tasks:
            # Tâches critiques : garde les providers premium
            if task.priority == 1 or task.requires_validation:
                optimized_tasks.append(task)
                continue
            
            # Tâches non-critiques : utilise providers économiques
            cost_effective_providers = self._get_cost_effective_providers(task.task_type)
            if cost_effective_providers:
                task.preferred_provider = cost_effective_providers[0]
                task.validation_providers = cost_effective_providers[1:2]  # 1 seul pour validation
            
            optimized_tasks.append(task)
        
        logger.info(
            "Tasks optimized for cost",
            original_count=len(tasks),
            optimized_count=len(optimized_tasks)
        )
        
        return optimized_tasks
    
    def _get_specialized_providers(self, task_type: LLMTaskType) -> List[LLMProvider]:
        """Récupère les providers spécialisés pour un type de tâche."""
        return self.TASK_SPECIALIZATIONS.get(task_type, [
            self.config.primary_provider
        ] + self.config.fallback_providers)
    
    def _is_provider_available(self, provider: LLMProvider) -> bool:
        """Vérifie si un provider est disponible."""
        if provider not in self.performance_tracker:
            return True  # Nouveau provider, assume disponible
        
        performance = self.performance_tracker[provider]
        return performance.is_available
    
    def _get_fallback_providers(self) -> List[LLMProvider]:
        """Récupère les providers de fallback."""
        fallbacks = [self.config.primary_provider] + self.config.fallback_providers
        
        # Filtre par disponibilité
        available_fallbacks = [
            provider for provider in fallbacks
            if self._is_provider_available(provider)
        ]
        
        if not available_fallbacks:
            # Dernier recours
            return [LLMProvider.FALLBACK]
        
        return available_fallbacks
    
    async def _select_optimal_provider(self, providers: List[LLMProvider], task: LLMTask) -> LLMProvider:
        """Sélectionne le provider optimal selon les critères."""
        if len(providers) == 1:
            return providers[0]
        
        # Si un provider préféré est spécifié et disponible
        if task.preferred_provider and task.preferred_provider in providers:
            return task.preferred_provider
        
        # Calcul des scores pour chaque provider
        provider_scores = {}
        
        for provider in providers:
            performance = self.performance_tracker.get(provider)
            if not performance:
                # Nouveau provider, score neutre
                provider_scores[provider] = 0.5
                continue
            
            # Score basé sur l'efficacité et la charge actuelle
            base_score = performance.efficiency_score
            
            # Pénalité pour charge élevée
            load_penalty = min(0.3, self.load_balancer.get(provider, 0) / 10)
            
            # Bonus pour coût faible si optimisation coût activée
            cost_bonus = 0.0
            if self.config.enable_cost_optimization:
                max_cost = max(self.PROVIDER_COSTS.values())
                cost_ratio = performance.cost_per_token / max_cost
                cost_bonus = (1.0 - cost_ratio) * 0.2
            
            final_score = base_score - load_penalty + cost_bonus
            provider_scores[provider] = final_score
        
        # Sélectionne le provider avec le meilleur score
        optimal_provider = max(provider_scores.items(), key=lambda x: x[1])[0]
        
        logger.debug(
            "Provider selection completed",
            provider_scores=provider_scores,
            selected=optimal_provider
        )
        
        return optimal_provider
    
    def _get_cost_effective_providers(self, task_type: LLMTaskType) -> List[LLMProvider]:
        """Récupère les providers les plus économiques pour une tâche."""
        specialized = self._get_specialized_providers(task_type)
        
        # Trie par coût croissant
        cost_sorted = sorted(
            specialized,
            key=lambda p: self.PROVIDER_COSTS.get(p, float('inf'))
        )
        
        # Garde les 3 moins chers
        return cost_sorted[:3]
    
    def _build_nutrition_prompt(self, product_data: Dict[str, Any]) -> str:
        """Construit le prompt pour analyse nutritionnelle."""
        return f"""
        Analysez le profil nutritionnel de ce produit:
        
        Produit: {product_data.get('product_name', 'Produit inconnu')}
        Ingrédients: {product_data.get('ingredients', [])}
        Valeurs nutritionnelles: {product_data.get('nutrition_facts', {})}
        
        Fournissez une analyse structurée en JSON avec:
        - health_score (0-10)
        - main_nutrients (liste)
        - nutritional_quality (excellent/good/average/poor)
        - recommendations (liste)
        """
    
    def _build_allergen_prompt(self, product_data: Dict[str, Any]) -> str:
        """Construit le prompt pour détection d'allergènes."""
        return f"""
        Analysez les allergènes dans ce produit:
        
        Ingrédients: {product_data.get('ingredients', [])}
        
        Identifiez en JSON:
        - allergens_detected (liste des 14 allergènes majeurs EU)
        - traces_possible (liste)
        - confidence_level (high/medium/low)
        - analysis_details (explications)
        """
    
    def _build_health_impact_prompt(self, product_data: Dict[str, Any]) -> str:
        """Construit le prompt pour évaluation impact santé."""
        return f"""
        Évaluez l'impact santé de ce produit:
        
        Produit: {product_data.get('product_name', 'Produit inconnu')}
        Nutrition: {product_data.get('nutrition_facts', {})}
        Ingrédients: {product_data.get('ingredients', [])}
        
        Fournissez en JSON:
        - health_impact_score (0-10)
        - positive_aspects (liste)
        - concerns (liste)
        - target_population (qui peut consommer)
        - consumption_advice (conseils)
        """
    
    def _build_categorization_prompt(self, product_data: Dict[str, Any]) -> str:
        """Construit le prompt pour catégorisation."""
        return f"""
        Catégorisez ce produit alimentaire:
        
        Nom: {product_data.get('product_name', 'Produit inconnu')}
        Catégorie actuelle: {product_data.get('category', 'Non spécifiée')}
        Ingrédients: {product_data.get('ingredients', [])}
        
        Retournez en JSON:
        - main_category (catégorie principale)
        - sub_categories (sous-catégories)
        - dietary_categories (vegan, vegetarian, etc.)
        - meal_type (breakfast, snack, etc.)
        """
    
    def _build_claims_validation_prompt(self, product_data: Dict[str, Any]) -> str:
        """Construit le prompt pour validation des claims."""
        claims = product_data.get('marketing_claims', []) + product_data.get('health_claims', [])
        
        return f"""
        Validez les affirmations de ce produit selon les réglementations EU/FDA:
        
        Produit: {product_data.get('product_name', 'Produit inconnu')}
        Affirmations: {claims}
        Nutrition: {product_data.get('nutrition_facts', {})}
        
        Analysez en JSON:
        - valid_claims (affirmations conformes)
        - invalid_claims (affirmations non conformes)
        - regulatory_warnings (avertissements)
        - suggested_modifications (modifications suggérées)
        """
    
    async def record_task_result(self, provider: LLMProvider, duration: float, 
                                quality_score: float, success: bool):
        """Enregistre le résultat d'une tâche pour améliorer le routage."""
        if provider in self.performance_tracker:
            self.performance_tracker[provider].record_request(duration, quality_score, success)
        
        # Décrémente le compteur de charge
        if provider in self.load_balancer:
            self.load_balancer[provider] = max(0, self.load_balancer[provider] - 1)
        
        logger.debug(
            "Task result recorded",
            provider=provider,
            duration=duration,
            quality_score=quality_score,
            success=success
        )