"""
Consensus Validator pour validation croisée entre LLMs.

Implémente la validation croisée entre plusieurs LLMs pour garantir
accuracy et cohérence des résultats d'analyse nutritionnelle.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog
import re
from difflib import SequenceMatcher

from .interfaces import (
    IConsensusValidator, LLMResult, LLMTask, LLMProvider, 
    LLMTaskType, QualityMetrics, ConfidenceLevel,
    LLMValidationException
)

logger = structlog.get_logger(__name__)


@dataclass
class ConsensusMetrics:
    """Métriques de consensus entre résultats LLM."""
    
    agreement_score: float  # 0-1, accord global
    structural_similarity: float  # Similarité structure JSON
    semantic_similarity: float  # Similarité sémantique
    factual_consistency: float  # Cohérence factuelle
    
    conflicting_fields: List[str]  # Champs en désaccord
    consensus_fields: List[str]   # Champs en accord
    
    primary_weight: float = 0.6   # Poids du résultat primaire
    secondary_weight: float = 0.3  # Poids du secondaire
    tertiary_weight: float = 0.1   # Poids du tertiaire
    
    @property
    def is_reliable_consensus(self) -> bool:
        """Indique si le consensus est fiable."""
        return (
            self.agreement_score >= 0.8 and
            self.factual_consistency >= 0.7 and
            len(self.conflicting_fields) <= 2
        )


class ConsensusValidator(IConsensusValidator):
    """
    Validateur de consensus pour résultats LLM multiples.
    
    Fonctionnalités:
    - Comparaison structurelle des résultats JSON
    - Analyse sémantique des différences
    - Résolution intelligente des conflits
    - Scoring de confiance du consensus
    - Détection d'hallucinations croisées
    """
    
    def __init__(self, min_providers: int = 2, consensus_threshold: float = 0.8):
        self.min_providers = min_providers
        self.consensus_threshold = consensus_threshold
        
        # Poids des champs pour résolution de conflits
        self.field_weights = {
            # Champs critiques (poids élevé)
            "health_score": 1.0,
            "allergens_detected": 1.0,
            "nutritional_quality": 0.9,
            "health_impact_score": 0.9,
            
            # Champs importants (poids moyen)
            "main_nutrients": 0.7,
            "positive_aspects": 0.6,
            "concerns": 0.6,
            "recommendations": 0.5,
            
            # Champs informatifs (poids faible)
            "analysis_details": 0.3,
            "consumption_advice": 0.3,
            "target_population": 0.4
        }
        
        logger.info(
            "Consensus Validator initialized",
            min_providers=min_providers,
            consensus_threshold=consensus_threshold
        )
    
    async def validate_result(self, primary_result: LLMResult, task: LLMTask) -> LLMResult:
        """
        Valide un résultat avec consensus multi-LLM.
        
        Args:
            primary_result: Résultat principal à valider
            task: Tâche originale pour context
            
        Returns:
            LLMResult: Résultat validé avec consensus
        """
        try:
            if not task.requires_validation:
                primary_result.is_validated = True
                primary_result.consensus_score = 1.0
                return primary_result
            
            # Obtient les résultats de validation
            validation_results = await self._get_validation_results(task)
            
            if not validation_results:
                logger.warning(
                    "No validation results available, using primary only",
                    task_type=task.task_type,
                    provider=primary_result.provider_used
                )
                primary_result.is_validated = False
                return primary_result
            
            # Calcule le consensus
            all_results = [primary_result] + validation_results
            consensus_metrics = await self._calculate_consensus_metrics(all_results)
            
            # Résout les conflits si nécessaire
            if not consensus_metrics.is_reliable_consensus:
                resolved_result = await self._resolve_conflicts(all_results, consensus_metrics)
            else:
                resolved_result = await self._merge_consensus_results(all_results, consensus_metrics)
            
            # Met à jour les métadonnées de validation
            resolved_result.is_validated = True
            resolved_result.validation_results = [
                {
                    "provider": result.provider_used,
                    "confidence": result.confidence_score,
                    "quality": result.quality_metrics.overall_score
                }
                for result in validation_results
            ]
            resolved_result.consensus_score = consensus_metrics.agreement_score
            
            logger.info(
                "Result validated successfully",
                consensus_score=consensus_metrics.agreement_score,
                conflicting_fields=len(consensus_metrics.conflicting_fields),
                providers_count=len(all_results)
            )
            
            return resolved_result
            
        except Exception as e:
            logger.error(
                "Validation failed",
                error=str(e),
                task_type=task.task_type
            )
            # Retourne le résultat primaire en cas d'erreur
            primary_result.is_validated = False
            return primary_result
    
    async def calculate_consensus_score(self, results: List[LLMResult]) -> float:
        """
        Calcule le score de consensus entre plusieurs résultats.
        
        Args:
            results: Liste des résultats à comparer
            
        Returns:
            float: Score de consensus (0-1)
        """
        if len(results) < 2:
            return 1.0
        
        try:
            consensus_metrics = await self._calculate_consensus_metrics(results)
            return consensus_metrics.agreement_score
            
        except Exception as e:
            logger.error("Consensus calculation failed", error=str(e))
            return 0.0
    
    async def resolve_conflicts(self, conflicting_results: List[LLMResult]) -> LLMResult:
        """
        Résout les conflits entre résultats divergents.
        
        Args:
            conflicting_results: Résultats en conflit
            
        Returns:
            LLMResult: Résultat consensus résolu
        """
        if len(conflicting_results) <= 1:
            return conflicting_results[0] if conflicting_results else None
        
        try:
            consensus_metrics = await self._calculate_consensus_metrics(conflicting_results)
            return await self._resolve_conflicts(conflicting_results, consensus_metrics)
            
        except Exception as e:
            logger.error("Conflict resolution failed", error=str(e))
            # Retourne le résultat avec la meilleure qualité
            return max(conflicting_results, key=lambda r: r.quality_metrics.overall_score)
    
    async def _get_validation_results(self, task: LLMTask) -> List[LLMResult]:
        """Obtient les résultats de validation depuis d'autres providers."""
        # Cette méthode sera appelée par le LLMServiceManager
        # qui a accès aux différents providers
        # Pour l'instant, retourne une liste vide
        # L'implémentation complète sera dans le manager
        return []
    
    async def _calculate_consensus_metrics(self, results: List[LLMResult]) -> ConsensusMetrics:
        """Calcule les métriques de consensus entre résultats."""
        if len(results) < 2:
            return ConsensusMetrics(
                agreement_score=1.0,
                structural_similarity=1.0,
                semantic_similarity=1.0,
                factual_consistency=1.0,
                conflicting_fields=[],
                consensus_fields=[]
            )
        
        # Parse les analyses JSON
        parsed_results = []
        for result in results:
            try:
                if isinstance(result.analysis, dict):
                    parsed_results.append(result.analysis)
                else:
                    parsed = json.loads(result.analysis)
                    parsed_results.append(parsed)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "Failed to parse result analysis",
                    provider=result.provider_used,
                    error=str(e)
                )
                continue
        
        if len(parsed_results) < 2:
            # Pas assez de résultats valides
            return ConsensusMetrics(
                agreement_score=0.5,
                structural_similarity=0.5,
                semantic_similarity=0.5,
                factual_consistency=0.5,
                conflicting_fields=[],
                consensus_fields=[]
            )
        
        # Calcul des métriques
        structural_sim = self._calculate_structural_similarity(parsed_results)
        semantic_sim = await self._calculate_semantic_similarity(parsed_results)
        factual_consistency = self._calculate_factual_consistency(parsed_results)
        
        # Identification des champs en conflit/accord
        conflicting_fields, consensus_fields = self._identify_field_agreements(parsed_results)
        
        # Score global d'accord
        agreement_score = (
            structural_sim * 0.3 +
            semantic_sim * 0.4 +
            factual_consistency * 0.3
        )
        
        return ConsensusMetrics(
            agreement_score=agreement_score,
            structural_similarity=structural_sim,
            semantic_similarity=semantic_sim,
            factual_consistency=factual_consistency,
            conflicting_fields=conflicting_fields,
            consensus_fields=consensus_fields
        )
    
    def _calculate_structural_similarity(self, parsed_results: List[Dict]) -> float:
        """Calcule la similarité structurelle des résultats JSON."""
        if len(parsed_results) < 2:
            return 1.0
        
        # Utilise le premier résultat comme référence
        reference = parsed_results[0]
        similarities = []
        
        for result in parsed_results[1:]:
            # Compare les clés présentes
            ref_keys = set(reference.keys())
            result_keys = set(result.keys())
            
            # Intersection et union des clés
            common_keys = ref_keys.intersection(result_keys)
            total_keys = ref_keys.union(result_keys)
            
            if not total_keys:
                similarities.append(1.0)
            else:
                # Jaccard similarity pour les clés
                key_similarity = len(common_keys) / len(total_keys)
                similarities.append(key_similarity)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    async def _calculate_semantic_similarity(self, parsed_results: List[Dict]) -> float:
        """Calcule la similarité sémantique entre résultats."""
        if len(parsed_results) < 2:
            return 1.0
        
        # Pour une implémentation simple, compare les valeurs textuelles
        # Dans une version avancée, utiliserait des embeddings
        similarities = []
        reference = parsed_results[0]
        
        for result in parsed_results[1:]:
            field_similarities = []
            
            for key in reference.keys():
                if key in result:
                    ref_value = str(reference[key]).lower()
                    result_value = str(result[key]).lower()
                    
                    # Utilise SequenceMatcher pour similarité textuelle
                    similarity = SequenceMatcher(None, ref_value, result_value).ratio()
                    field_similarities.append(similarity)
            
            if field_similarities:
                avg_similarity = sum(field_similarities) / len(field_similarities)
                similarities.append(avg_similarity)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def _calculate_factual_consistency(self, parsed_results: List[Dict]) -> float:
        """Calcule la cohérence factuelle entre résultats."""
        if len(parsed_results) < 2:
            return 1.0
        
        # Vérifie la cohérence sur des champs factuels critiques
        critical_fields = [
            "health_score", "allergens_detected", "nutritional_quality",
            "health_impact_score"
        ]
        
        consistencies = []
        
        for field in critical_fields:
            field_values = []
            for result in parsed_results:
                if field in result and result[field] is not None:
                    field_values.append(result[field])
            
            if len(field_values) < 2:
                continue
            
            # Pour scores numériques
            if field.endswith("_score") and all(isinstance(v, (int, float)) for v in field_values):
                # Calcule variance relative
                mean_val = sum(field_values) / len(field_values)
                if mean_val > 0:
                    variance = sum((v - mean_val) ** 2 for v in field_values) / len(field_values)
                    relative_variance = variance / (mean_val ** 2)
                    consistency = max(0.0, 1.0 - relative_variance)
                    consistencies.append(consistency)
            
            # Pour listes (allergènes, etc.)
            elif isinstance(field_values[0], list):
                # Compare les intersections
                sets = [set(v) if isinstance(v, list) else {v} for v in field_values]
                intersection = sets[0]
                union = sets[0]
                
                for s in sets[1:]:
                    intersection = intersection.intersection(s)
                    union = union.union(s)
                
                if union:
                    jaccard = len(intersection) / len(union)
                    consistencies.append(jaccard)
            
            # Pour valeurs catégorielles
            else:
                unique_values = set(field_values)
                if len(unique_values) == 1:
                    consistencies.append(1.0)  # Tous identiques
                else:
                    # Accord partiel basé sur la proportion majoritaire
                    value_counts = {v: field_values.count(v) for v in unique_values}
                    max_count = max(value_counts.values())
                    agreement = max_count / len(field_values)
                    consistencies.append(agreement)
        
        return sum(consistencies) / len(consistencies) if consistencies else 1.0
    
    def _identify_field_agreements(self, parsed_results: List[Dict]) -> Tuple[List[str], List[str]]:
        """Identifie les champs en conflit et en accord."""
        if len(parsed_results) < 2:
            return [], list(parsed_results[0].keys()) if parsed_results else []
        
        all_fields = set()
        for result in parsed_results:
            all_fields.update(result.keys())
        
        conflicting_fields = []
        consensus_fields = []
        
        for field in all_fields:
            field_values = []
            for result in parsed_results:
                if field in result:
                    field_values.append(result[field])
            
            if len(field_values) < 2:
                continue
            
            # Détermine s'il y a conflit
            if field.endswith("_score") and all(isinstance(v, (int, float)) for v in field_values):
                # Pour scores numériques, vérifie la variance
                mean_val = sum(field_values) / len(field_values)
                variance = sum((v - mean_val) ** 2 for v in field_values) / len(field_values)
                if variance > (mean_val * 0.2) ** 2:  # > 20% de variance relative
                    conflicting_fields.append(field)
                else:
                    consensus_fields.append(field)
            
            elif isinstance(field_values[0], list):
                # Pour listes, vérifie la cohérence
                sets = [set(v) if isinstance(v, list) else {v} for v in field_values]
                intersection = sets[0]
                for s in sets[1:]:
                    intersection = intersection.intersection(s)
                
                if len(intersection) < len(sets[0]) * 0.5:  # < 50% d'accord
                    conflicting_fields.append(field)
                else:
                    consensus_fields.append(field)
            
            else:
                # Pour autres types, vérifie l'unicité
                unique_values = set(str(v).lower() for v in field_values)
                if len(unique_values) > 1:
                    conflicting_fields.append(field)
                else:
                    consensus_fields.append(field)
        
        return conflicting_fields, consensus_fields
    
    async def _resolve_conflicts(self, results: List[LLMResult], 
                                consensus_metrics: ConsensusMetrics) -> LLMResult:
        """Résout les conflits entre résultats divergents."""
        if not results:
            raise LLMValidationException("No results to resolve conflicts")
        
        # Trie les résultats par qualité
        sorted_results = sorted(
            results, 
            key=lambda r: r.quality_metrics.overall_score, 
            reverse=True
        )
        
        # Utilise le meilleur résultat comme base
        base_result = sorted_results[0]
        base_analysis = base_result.analysis
        
        if isinstance(base_analysis, dict):
            resolved_analysis = base_analysis.copy()
        else:
            try:
                resolved_analysis = json.loads(base_analysis)
            except json.JSONDecodeError:
                # Si parsing échoue, retourne le meilleur résultat tel quel
                return base_result
        
        # Résout chaque champ en conflit
        for field in consensus_metrics.conflicting_fields:
            field_values = []
            field_weights = []
            
            for result in results:
                try:
                    if isinstance(result.analysis, dict):
                        analysis = result.analysis
                    else:
                        analysis = json.loads(result.analysis)
                    
                    if field in analysis:
                        field_values.append(analysis[field])
                        # Poids basé sur qualité du résultat
                        weight = result.quality_metrics.overall_score
                        field_weights.append(weight)
                
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if field_values and field_weights:
                resolved_value = self._resolve_field_conflict(
                    field, field_values, field_weights
                )
                resolved_analysis[field] = resolved_value
        
        # Crée le résultat résolu
        resolved_result = LLMResult(
            analysis=resolved_analysis,
            raw_response=f"Consensus from {len(results)} providers",
            quality_metrics=self._calculate_consensus_quality_metrics(results),
            confidence_score=consensus_metrics.agreement_score,
            provider_used=LLMProvider.FALLBACK,  # Indique consensus
            task_type=base_result.task_type,
            processing_time=sum(r.processing_time for r in results) / len(results)
        )
        
        return resolved_result
    
    async def _merge_consensus_results(self, results: List[LLMResult], 
                                     consensus_metrics: ConsensusMetrics) -> LLMResult:
        """Merge les résultats en consensus quand ils sont cohérents."""
        if not results:
            raise LLMValidationException("No results to merge")
        
        # Utilise le résultat avec la meilleure qualité comme base
        best_result = max(results, key=lambda r: r.quality_metrics.overall_score)
        
        # Améliore la confiance grâce au consensus
        enhanced_confidence = min(1.0, best_result.confidence_score * 1.2)
        
        # Améliore les métriques qualité
        enhanced_quality = self._calculate_consensus_quality_metrics(results)
        
        # Crée le résultat amélioré
        consensus_result = LLMResult(
            analysis=best_result.analysis,
            raw_response=best_result.raw_response,
            quality_metrics=enhanced_quality,
            confidence_score=enhanced_confidence,
            provider_used=best_result.provider_used,
            task_type=best_result.task_type,
            processing_time=best_result.processing_time
        )
        
        return consensus_result
    
    def _resolve_field_conflict(self, field: str, values: List[Any], weights: List[float]) -> Any:
        """Résout un conflit sur un champ spécifique."""
        if not values:
            return None
        
        # Poids du champ
        field_weight = self.field_weights.get(field, 0.5)
        
        # Pour scores numériques : moyenne pondérée
        if field.endswith("_score") and all(isinstance(v, (int, float)) for v in values):
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else values[0]
        
        # Pour listes : union avec préférence aux plus fréquents
        elif all(isinstance(v, list) for v in values):
            all_items = []
            for v, w in zip(values, weights):
                # Ajoute chaque item avec son poids
                for item in v:
                    all_items.extend([item] * int(w * 10))  # Multiplie par 10 pour éviter les décimales
            
            # Compte les occurrences
            item_counts = {}
            for item in all_items:
                item_counts[item] = item_counts.get(item, 0) + 1
            
            # Garde les items les plus fréquents
            sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
            threshold = max(1, len(values) * 0.5)  # Au moins 50% des providers
            
            consensus_items = [item for item, count in sorted_items if count >= threshold]
            return consensus_items
        
        # Pour valeurs catégorielles : vote pondéré
        else:
            value_weights = {}
            for v, w in zip(values, weights):
                v_str = str(v).lower()
                value_weights[v_str] = value_weights.get(v_str, 0) + w
            
            # Sélectionne la valeur avec le poids le plus élevé
            best_value = max(value_weights.items(), key=lambda x: x[1])[0]
            
            # Trouve la valeur originale correspondante
            for v in values:
                if str(v).lower() == best_value:
                    return v
            
            return values[0]  # Fallback
    
    def _calculate_consensus_quality_metrics(self, results: List[LLMResult]) -> QualityMetrics:
        """Calcule les métriques qualité du consensus."""
        if not results:
            return QualityMetrics(
                data_completeness=0.0,
                logical_consistency=0.0,
                source_citation=0.0,
                confidence_calibration=0.0,
                format_compliance=0.0
            )
        
        # Moyenne pondérée des métriques
        total_weight = sum(r.quality_metrics.overall_score for r in results)
        
        if total_weight == 0:
            # Tous les résultats ont une qualité nulle, utilise moyenne simple
            avg_completeness = sum(r.quality_metrics.data_completeness for r in results) / len(results)
            avg_consistency = sum(r.quality_metrics.logical_consistency for r in results) / len(results)
            avg_citation = sum(r.quality_metrics.source_citation for r in results) / len(results)
            avg_calibration = sum(r.quality_metrics.confidence_calibration for r in results) / len(results)
            avg_compliance = sum(r.quality_metrics.format_compliance for r in results) / len(results)
        else:
            # Moyenne pondérée par qualité
            avg_completeness = sum(
                r.quality_metrics.data_completeness * r.quality_metrics.overall_score 
                for r in results
            ) / total_weight
            
            avg_consistency = sum(
                r.quality_metrics.logical_consistency * r.quality_metrics.overall_score 
                for r in results
            ) / total_weight
            
            avg_citation = sum(
                r.quality_metrics.source_citation * r.quality_metrics.overall_score 
                for r in results
            ) / total_weight
            
            avg_calibration = sum(
                r.quality_metrics.confidence_calibration * r.quality_metrics.overall_score 
                for r in results
            ) / total_weight
            
            avg_compliance = sum(
                r.quality_metrics.format_compliance * r.quality_metrics.overall_score 
                for r in results
            ) / total_weight
        
        # Bonus pour consensus multiple (plus de providers = plus fiable)
        consensus_bonus = min(0.1, (len(results) - 1) * 0.05)
        
        return QualityMetrics(
            data_completeness=min(1.0, avg_completeness + consensus_bonus),
            logical_consistency=min(1.0, avg_consistency + consensus_bonus),
            source_citation=avg_citation,
            confidence_calibration=min(1.0, avg_calibration + consensus_bonus),
            format_compliance=avg_compliance,
            analysis_duration=sum(r.processing_time for r in results),
            token_usage=sum(getattr(r.quality_metrics, 'token_usage', 0) for r in results),
            provider_used=f"Consensus({len(results)}providers)"
        )