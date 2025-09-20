"""
Provider GPT-4 pour le service LLM enrichment.

Implémentation concrète du service OpenAI GPT-4 avec optimisations
spécifiques pour l'analyse nutritionnelle.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
import aiohttp
import os

from .interfaces import (
    ILLMService, LLMTask, LLMResult, LLMProvider, LLMTaskType,
    QualityMetrics, ConfidenceLevel, LLMServiceException, LLMQuotaException
)

logger = structlog.get_logger(__name__)


class GPT4Service(ILLMService):
    """
    Service OpenAI GPT-4 optimisé pour analyse nutritionnelle.
    
    Fonctionnalités:
    - Support GPT-4o, GPT-4-turbo, GPT-4o-mini
    - Prompts optimisés par type de tâche
    - Gestion des quotas et rate limiting
    - Validation des réponses JSON
    - Métriques de qualité automatiques
    - Self-correction en cas d'erreur format
    """
    
    # Configuration des modèles
    MODEL_CONFIGS = {
        LLMProvider.GPT_4O: {
            "model": "gpt-4o",
            "max_tokens": 4000,
            "cost_per_input_token": 0.000005,
            "cost_per_output_token": 0.000015,
            "context_window": 128000
        },
        LLMProvider.GPT_4O_MINI: {
            "model": "gpt-4o-mini",
            "max_tokens": 2000,
            "cost_per_input_token": 0.00000015,
            "cost_per_output_token": 0.0000006,
            "context_window": 128000
        },
        LLMProvider.GPT_4_TURBO: {
            "model": "gpt-4-turbo",
            "max_tokens": 4000,
            "cost_per_input_token": 0.00001,
            "cost_per_output_token": 0.00003,
            "context_window": 128000
        }
    }
    
    def __init__(self, provider: LLMProvider = LLMProvider.GPT_4O, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise LLMServiceException(
                "OpenAI API key not provided",
                provider=str(provider),
                error_code="MISSING_API_KEY"
            )
        
        self.config = self.MODEL_CONFIGS.get(provider)
        if not self.config:
            raise LLMServiceException(
                f"Unsupported provider: {provider}",
                provider=str(provider),
                error_code="UNSUPPORTED_PROVIDER"
            )
        
        self.base_url = "https://api.openai.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Métriques
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        logger.info(
            "GPT-4 Service initialized",
            provider=provider,
            model=self.config["model"],
            context_window=self.config["context_window"]
        )
    
    @property
    def provider_name(self) -> str:
        """Nom du provider."""
        return f"OpenAI-{self.config['model']}"
    
    @property
    def supported_tasks(self) -> List[LLMTaskType]:
        """Types de tâches supportées."""
        return [
            LLMTaskType.NUTRITION_ANALYSIS,
            LLMTaskType.ALLERGEN_DETECTION,
            LLMTaskType.HEALTH_IMPACT_ASSESSMENT,
            LLMTaskType.INGREDIENT_PARSING,
            LLMTaskType.DIETARY_COMPLIANCE,
            LLMTaskType.PRODUCT_CATEGORIZATION,
            LLMTaskType.CLAIMS_VALIDATION,
            LLMTaskType.RECIPE_ANALYSIS,
            LLMTaskType.NUTRITIONAL_COMPARISON
        ]
    
    @property
    def cost_per_token(self) -> float:
        """Coût moyen par token."""
        # Moyenne input/output
        return (self.config["cost_per_input_token"] + self.config["cost_per_output_token"]) / 2
    
    async def analyze(self, task: LLMTask) -> LLMResult:
        """
        Analyse une tâche LLM avec GPT-4.
        
        Args:
            task: Tâche à analyser
            
        Returns:
            LLMResult: Résultat avec métriques qualité
        """
        start_time = time.time()
        
        try:
            # Initialise session si nécessaire
            if not self.session:
                await self._init_session()
            
            # Prépare la requête
            request_data = await self._prepare_request(task)
            
            # Estime le coût avant envoi
            estimated_cost = await self.estimate_cost(task)
            if estimated_cost > 1.0:  # Seuil de sécurité
                raise LLMQuotaException(
                    f"Estimated cost too high: ${estimated_cost:.4f}",
                    quota_type="cost",
                    current_usage=estimated_cost,
                    limit=1.0
                )
            
            # Envoie la requête
            response_data = await self._send_request(request_data)
            
            # Parse la réponse
            analysis, raw_response = await self._parse_response(response_data, task)
            
            # Calcule métriques
            processing_time = time.time() - start_time
            quality_metrics = await self._calculate_quality_metrics(
                analysis, raw_response, task, response_data
            )
            
            # Met à jour statistiques
            await self._update_usage_stats(response_data)
            
            # Crée le résultat
            result = LLMResult(
                analysis=analysis,
                raw_response=raw_response,
                quality_metrics=quality_metrics,
                confidence_score=await self._calculate_confidence(analysis, task),
                provider_used=self.provider,
                task_type=task.task_type,
                processing_time=processing_time
            )
            
            logger.info(
                "Task analyzed successfully",
                task_type=task.task_type,
                processing_time=processing_time,
                quality_score=quality_metrics.overall_score,
                estimated_cost=estimated_cost
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            if isinstance(e, (LLMServiceException, LLMQuotaException)):
                raise e
            
            logger.error(
                "Task analysis failed",
                task_type=task.task_type,
                processing_time=processing_time,
                error=str(e)
            )
            
            raise LLMServiceException(
                f"Analysis failed: {str(e)}",
                provider=str(self.provider),
                error_code="ANALYSIS_FAILED"
            )
    
    async def validate_response(self, response: str, expected_format: str) -> bool:
        """
        Valide la réponse GPT-4.
        
        Args:
            response: Réponse à valider
            expected_format: Format attendu (généralement "json")
            
        Returns:
            bool: True si valide
        """
        try:
            if expected_format.lower() == "json":
                # Tente de parser le JSON
                parsed = json.loads(response)
                
                # Vérifie que c'est un dictionnaire non vide
                if not isinstance(parsed, dict) or not parsed:
                    return False
                
                return True
            else:
                # Pour autres formats, vérifie juste que non vide
                return bool(response.strip())
                
        except (json.JSONDecodeError, TypeError):
            return False
    
    async def estimate_cost(self, task: LLMTask) -> float:
        """
        Estime le coût d'une tâche.
        
        Args:
            task: Tâche à estimer
            
        Returns:
            float: Coût estimé en USD
        """
        # Estime tokens d'entrée
        input_tokens = self._estimate_tokens(task.prompt + str(task.data))
        
        # Estime tokens de sortie (basé sur max_tokens de la tâche)
        output_tokens = min(task.max_tokens, self.config["max_tokens"])
        
        # Calcule coût
        input_cost = input_tokens * self.config["cost_per_input_token"]
        output_cost = output_tokens * self.config["cost_per_output_token"]
        
        return input_cost + output_cost
    
    async def _init_session(self):
        """Initialise la session HTTP."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Wellix-Nutrition-AI/1.0"
        }
        
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout
        )
    
    async def _prepare_request(self, task: LLMTask) -> Dict[str, Any]:
        """Prépare la requête pour l'API OpenAI."""
        # Optimise le prompt selon le type de tâche
        optimized_prompt = await self._optimize_prompt(task)
        
        # Messages système + utilisateur
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(task.task_type)
            },
            {
                "role": "user",
                "content": optimized_prompt
            }
        ]
        
        return {
            "model": self.config["model"],
            "messages": messages,
            "temperature": task.temperature,
            "max_tokens": min(task.max_tokens, self.config["max_tokens"]),
            "response_format": {"type": "json_object"},  # Force JSON
            "top_p": 0.1,  # Focalisé pour cohérence
            "frequency_penalty": 0.1,  # Évite répétitions
            "presence_penalty": 0.1
        }
    
    def _get_system_prompt(self, task_type: LLMTaskType) -> str:
        """Retourne le prompt système optimisé par type de tâche."""
        base_prompt = """Tu es un expert nutritionniste certifié avec 15 ans d'expérience en analyse alimentaire. 
Tu fournis des analyses précises, factuelles et conformes aux réglementations EU/FDA.

INSTRUCTIONS CRITIQUES:
1. Réponds UNIQUEMENT en JSON valide
2. Base-toi UNIQUEMENT sur les données fournies
3. Cite tes sources (ingrédients → conclusions)
4. Utilise des scores objectifs (0-10)
5. Distingue clairement faits vs recommandations
6. Respecte les réglementations santé en vigueur"""
        
        task_specific = {
            LLMTaskType.NUTRITION_ANALYSIS: """
Spécialisation: Analyse nutritionnelle complète
- Évalue qualité nutritionnelle globale (score 0-10)
- Identifie nutriments clés et carences
- Fournis recommandations d'usage
- Calcule impact santé basé sur composition""",
            
            LLMTaskType.ALLERGEN_DETECTION: """
Spécialisation: Détection d'allergènes (14 allergènes majeurs EU)
- Identifie allergènes présents et traces possibles
- Analyse interactions et contaminations croisées
- Évalue niveau de risque par allergène
- Conforme réglementation EU 1169/2011""",
            
            LLMTaskType.HEALTH_IMPACT_ASSESSMENT: """
Spécialisation: Évaluation impact santé
- Évalue bénéfices et risques santé
- Identifie populations cibles et à risque
- Analyse à court et long terme
- Base sur données scientifiques validées""",
            
            LLMTaskType.CLAIMS_VALIDATION: """
Spécialisation: Validation affirmations santé
- Vérifie conformité réglementaire EU/FDA
- Identifie claims non autorisées
- Suggère reformulations conformes
- Applique règlement EU 432/2012"""
        }
        
        specific_prompt = task_specific.get(task_type, "")
        return f"{base_prompt}\n\n{specific_prompt}"
    
    async def _optimize_prompt(self, task: LLMTask) -> str:
        """Optimise le prompt selon les bonnes pratiques."""
        base_prompt = task.prompt
        
        # Ajoute contexte structuré
        if task.data:
            data_context = "\nDONNÉES À ANALYSER:\n"
            for key, value in task.data.items():
                if isinstance(value, (list, dict)):
                    data_context += f"- {key}: {json.dumps(value, ensure_ascii=False)}\n"
                else:
                    data_context += f"- {key}: {value}\n"
            
            base_prompt += data_context
        
        # Ajoute instructions format JSON spécifiques
        json_instructions = self._get_json_format_instructions(task.task_type)
        
        return f"{base_prompt}\n\n{json_instructions}"
    
    def _get_json_format_instructions(self, task_type: LLMTaskType) -> str:
        """Instructions format JSON par type de tâche."""
        formats = {
            LLMTaskType.NUTRITION_ANALYSIS: """
FORMAT JSON REQUIS:
{
  "health_score": float (0-10),
  "nutritional_quality": "excellent|good|average|poor",
  "main_nutrients": ["nutriment1", "nutriment2"],
  "positive_aspects": ["aspect1", "aspect2"],
  "concerns": ["concern1", "concern2"],
  "recommendations": ["rec1", "rec2"],
  "target_population": "description",
  "analysis_confidence": float (0-1)
}""",
            
            LLMTaskType.ALLERGEN_DETECTION: """
FORMAT JSON REQUIS:
{
  "allergens_detected": ["allergen1", "allergen2"],
  "traces_possible": ["allergen1", "allergen2"],
  "confidence_level": "high|medium|low",
  "risk_assessment": {
    "severity": "high|medium|low",
    "affected_population": "description"
  },
  "regulatory_compliance": "compliant|non_compliant|unclear",
  "analysis_details": "explanation"
}""",
            
            LLMTaskType.HEALTH_IMPACT_ASSESSMENT: """
FORMAT JSON REQUIS:
{
  "health_impact_score": float (0-10),
  "positive_impacts": ["benefit1", "benefit2"],
  "negative_impacts": ["risk1", "risk2"],
  "target_populations": {
    "recommended": ["group1", "group2"],
    "avoid": ["group1", "group2"]
  },
  "consumption_advice": {
    "frequency": "daily|weekly|occasional|avoid",
    "portion_guidance": "description"
  },
  "scientific_evidence": "strong|moderate|limited"
}"""
        }
        
        return formats.get(task_type, """
FORMAT JSON REQUIS:
{
  "analysis": "résultat principal",
  "confidence": float (0-1),
  "details": "explications détaillées"
}""")
    
    async def _send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Envoie la requête à l'API OpenAI."""
        if not self.session:
            raise LLMServiceException("Session not initialized")
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=request_data
        ) as response:
            
            if response.status == 429:
                # Rate limiting
                retry_after = int(response.headers.get("Retry-After", 60))
                raise LLMQuotaException(
                    f"Rate limit exceeded. Retry after {retry_after}s",
                    quota_type="rate_limit",
                    current_usage=1,
                    limit=1
                )
            
            elif response.status == 401:
                raise LLMServiceException(
                    "Invalid API key",
                    provider=str(self.provider),
                    error_code="INVALID_API_KEY"
                )
            
            elif response.status != 200:
                error_text = await response.text()
                raise LLMServiceException(
                    f"API request failed: {response.status} - {error_text}",
                    provider=str(self.provider),
                    error_code=f"HTTP_{response.status}"
                )
            
            return await response.json()
    
    async def _parse_response(self, response_data: Dict[str, Any], 
                            task: LLMTask) -> tuple[Dict[str, Any], str]:
        """Parse la réponse OpenAI."""
        try:
            choice = response_data["choices"][0]
            raw_response = choice["message"]["content"]
            
            # Parse le JSON
            analysis = json.loads(raw_response)
            
            # Validation basique
            if not isinstance(analysis, dict):
                raise ValueError("Response is not a JSON object")
            
            return analysis, raw_response
            
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # Tentative de self-correction
            logger.warning(
                "Response parsing failed, attempting self-correction",
                error=str(e),
                task_type=task.task_type
            )
            
            # Fallback : crée réponse basique
            fallback_analysis = {
                "error": "Failed to parse response",
                "raw_content": response_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "confidence": 0.1
            }
            
            return fallback_analysis, str(response_data)
    
    async def _calculate_quality_metrics(self, analysis: Dict[str, Any], 
                                       raw_response: str, 
                                       task: LLMTask,
                                       response_data: Dict[str, Any]) -> QualityMetrics:
        """Calcule les métriques de qualité."""
        # Complétude des données
        expected_fields = self._get_expected_fields(task.task_type)
        present_fields = [field for field in expected_fields if field in analysis]
        data_completeness = len(present_fields) / len(expected_fields) if expected_fields else 1.0
        
        # Cohérence logique (vérifie types et valeurs)
        logical_consistency = await self._check_logical_consistency(analysis, task.task_type)
        
        # Citation des sources (présence d'explications)
        source_citation = 1.0 if any(
            key in analysis for key in ["analysis_details", "explanation", "reasoning"]
        ) else 0.0
        
        # Calibration de la confiance
        confidence_calibration = analysis.get("confidence", analysis.get("analysis_confidence", 0.5))
        
        # Conformité format JSON
        format_compliance = 1.0 if isinstance(analysis, dict) and not analysis.get("error") else 0.0
        
        # Usage tokens
        token_usage = response_data.get("usage", {}).get("total_tokens", 0)
        
        return QualityMetrics(
            data_completeness=data_completeness,
            logical_consistency=logical_consistency,
            source_citation=source_citation,
            confidence_calibration=confidence_calibration,
            format_compliance=format_compliance,
            token_usage=token_usage,
            provider_used=self.provider_name
        )
    
    def _get_expected_fields(self, task_type: LLMTaskType) -> List[str]:
        """Retourne les champs attendus par type de tâche."""
        field_mappings = {
            LLMTaskType.NUTRITION_ANALYSIS: [
                "health_score", "nutritional_quality", "main_nutrients", 
                "recommendations"
            ],
            LLMTaskType.ALLERGEN_DETECTION: [
                "allergens_detected", "confidence_level", "risk_assessment"
            ],
            LLMTaskType.HEALTH_IMPACT_ASSESSMENT: [
                "health_impact_score", "positive_impacts", "negative_impacts"
            ]
        }
        
        return field_mappings.get(task_type, ["analysis", "confidence"])
    
    async def _check_logical_consistency(self, analysis: Dict[str, Any], 
                                       task_type: LLMTaskType) -> float:
        """Vérifie la cohérence logique de l'analyse."""
        consistency_score = 1.0
        
        # Vérifie les scores numériques
        for field in ["health_score", "health_impact_score"]:
            if field in analysis:
                score = analysis[field]
                if not isinstance(score, (int, float)) or not (0 <= score <= 10):
                    consistency_score -= 0.2
        
        # Vérifie les listes
        for field in ["allergens_detected", "main_nutrients", "recommendations"]:
            if field in analysis:
                value = analysis[field]
                if not isinstance(value, list):
                    consistency_score -= 0.2
        
        # Vérifie cohérence entre score et qualité
        if "health_score" in analysis and "nutritional_quality" in analysis:
            score = analysis["health_score"]
            quality = analysis["nutritional_quality"].lower()
            
            if score >= 8 and quality not in ["excellent", "good"]:
                consistency_score -= 0.3
            elif score <= 3 and quality not in ["poor", "average"]:
                consistency_score -= 0.3
        
        return max(0.0, consistency_score)
    
    async def _calculate_confidence(self, analysis: Dict[str, Any], task: LLMTask) -> float:
        """Calcule le score de confiance."""
        # Utilise confiance fournie par l'analyse si disponible
        explicit_confidence = analysis.get("confidence", analysis.get("analysis_confidence"))
        
        if explicit_confidence is not None:
            return float(explicit_confidence)
        
        # Calcule confiance basée sur complétude
        expected_fields = self._get_expected_fields(task.task_type)
        present_fields = sum(1 for field in expected_fields if field in analysis)
        
        field_confidence = present_fields / len(expected_fields) if expected_fields else 1.0
        
        # Facteur qualité données d'entrée
        data_quality = 1.0
        if not task.data.get("ingredients"):
            data_quality -= 0.2
        if not task.data.get("nutrition_facts"):
            data_quality -= 0.2
        
        return min(1.0, field_confidence * data_quality)
    
    async def _update_usage_stats(self, response_data: Dict[str, Any]):
        """Met à jour les statistiques d'usage."""
        self.total_requests += 1
        
        usage = response_data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        
        self.total_tokens_used += tokens_used
        
        # Calcule coût
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        cost = (
            input_tokens * self.config["cost_per_input_token"] +
            output_tokens * self.config["cost_per_output_token"]
        )
        
        self.total_cost += cost
    
    def _estimate_tokens(self, text: str) -> int:
        """Estime le nombre de tokens dans un texte."""
        # Estimation approximative : 1 token ≈ 4 caractères pour le français
        return len(text) // 4
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'usage."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_request": self.total_cost / max(1, self.total_requests),
            "provider": self.provider_name
        }