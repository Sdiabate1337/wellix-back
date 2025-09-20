"""
Nodes finaux du workflow d'analyse nutritionnelle.
Implements la génération d'alternatives et l'assemblage de réponse.

Architecture Pattern : Strategy + Builder + Template Method
Inspiration : Builder Pattern, Fluent API, Chain of Responsibility
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import replace
from datetime import datetime
import structlog

from app.workflows.interfaces import (
    IWorkflowNode, WorkflowState, WorkflowStage, WorkflowNodeError,
    InputData, AnalysisConfig
)
from app.workflows.nodes.core_nodes import BaseWorkflowNode
from app.models.clinical import NutritionData
from app.models.health import UserHealthContext

logger = structlog.get_logger(__name__)


class AlternativeGenerationNode(BaseWorkflowNode):
    """
    Node de génération d'alternatives alimentaires.
    
    Responsabilités :
    - Analyser les défauts du produit actuel
    - Générer des alternatives plus saines
    - Prioriser selon le profil utilisateur
    - Fournir des justifications détaillées
    """
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.ALTERNATIVE_GENERATION
    
    async def _execute_business_logic(self, state: WorkflowState) -> WorkflowState:
        """Génère des alternatives alimentaires."""
        if not state.nutrition_data or not state.calculated_score:
            raise WorkflowNodeError(
                "Missing nutrition data or score for alternative generation",
                self.required_stage.value,
                recoverable=False
            )
        
        # Analyse des problèmes du produit actuel
        product_issues = await self._analyze_product_issues(
            state.nutrition_data,
            state.user_context,
            state.metadata.get("scores", {}),
            state.metadata.get("risk_assessment", {})
        )
        
        # Génération d'alternatives
        alternatives = await self._generate_alternatives(
            state.nutrition_data,
            state.user_context,
            product_issues,
            state.analysis_config
        )
        
        # Priorisation des alternatives
        prioritized_alternatives = await self._prioritize_alternatives(
            alternatives,
            state.user_context,
            product_issues
        )
        
        return replace(
            state,
            alternatives=prioritized_alternatives,
            metadata={
                **state.metadata,
                "product_issues": product_issues,
                "alternative_generation_timestamp": datetime.utcnow().isoformat(),
                "alternatives_count": len(prioritized_alternatives)
            }
        )
    
    async def _analyze_product_issues(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        scores: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyse les problèmes du produit actuel."""
        await asyncio.sleep(0.05)  # Simulation
        
        issues = []
        
        # Analyse des scores faibles
        nutri_score = scores.get("nutri_score", {})
        if nutri_score.get("letter") in ["D", "E"]:
            issues.append({
                "type": "poor_nutri_score",
                "severity": "high",
                "description": f"Poor Nutri-Score ({nutri_score.get('letter')})",
                "suggestions": ["Look for products with A or B rating"]
            })
        
        # Analyse des macronutriments
        if nutrition_data.sugar and nutrition_data.sugar > 20:
            issues.append({
                "type": "high_sugar",
                "severity": "medium",
                "value": nutrition_data.sugar,
                "description": f"High sugar content ({nutrition_data.sugar}g per 100g)",
                "suggestions": ["Choose products with <10g sugar per 100g"]
            })
        
        if nutrition_data.sodium and nutrition_data.sodium > 1.5:
            issues.append({
                "type": "high_sodium",
                "severity": "medium",
                "value": nutrition_data.sodium,
                "description": f"High sodium content ({nutrition_data.sodium}g per 100g)",
                "suggestions": ["Choose products with <0.3g sodium per 100g"]
            })
        
        if nutrition_data.total_fat and nutrition_data.total_fat > 20:
            issues.append({
                "type": "high_fat",
                "severity": "low",
                "value": nutrition_data.total_fat,
                "description": f"High fat content ({nutrition_data.total_fat}g per 100g)",
                "suggestions": ["Look for lower-fat alternatives"]
            })
        
        # Analyse des carences
        if not nutrition_data.fiber or nutrition_data.fiber < 3:
            issues.append({
                "type": "low_fiber",
                "severity": "medium",
                "value": nutrition_data.fiber or 0,
                "description": f"Low fiber content ({nutrition_data.fiber or 0}g per 100g)",
                "suggestions": ["Choose products with >6g fiber per 100g"]
            })
        
        # Analyse des risques santé personnalisés
        for risk in risk_assessment.get("risks", []):
            issues.append({
                "type": "health_risk",
                "severity": risk["severity"],
                "description": risk["message"],
                "suggestions": [f"Avoid due to {risk['type']}"]
            })
        
        return issues
    
    async def _generate_alternatives(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]],
        config: AnalysisConfig
    ) -> List[Dict[str, Any]]:
        """Génère des alternatives alimentaires."""
        await asyncio.sleep(0.1)  # Simulation d'algorithme de recherche
        
        alternatives = []
        
        # Stratégies de génération basées sur le type de produit
        product_category = self._categorize_product(nutrition_data)
        
        if product_category == "breakfast_spread":
            alternatives.extend(await self._generate_breakfast_spread_alternatives(
                nutrition_data, user_context, issues
            ))
        elif product_category == "snack":
            alternatives.extend(await self._generate_snack_alternatives(
                nutrition_data, user_context, issues
            ))
        elif product_category == "beverage":
            alternatives.extend(await self._generate_beverage_alternatives(
                nutrition_data, user_context, issues
            ))
        else:
            # Alternatives génériques
            alternatives.extend(await self._generate_generic_alternatives(
                nutrition_data, user_context, issues
            ))
        
        # Ajout d'alternatives basées sur les contraintes santé
        if user_context.health_conditions:
            health_alternatives = await self._generate_health_specific_alternatives(
                nutrition_data, user_context, issues
            )
            alternatives.extend(health_alternatives)
        
        return alternatives
    
    def _categorize_product(self, nutrition_data: NutritionData) -> str:
        """Catégorise le produit pour choisir la stratégie d'alternatives."""
        product_name = (nutrition_data.product_name or "").lower()
        
        if any(word in product_name for word in ["nutella", "spread", "jam", "honey"]):
            return "breakfast_spread"
        elif any(word in product_name for word in ["cookie", "biscuit", "chocolate", "candy"]):
            return "snack"
        elif any(word in product_name for word in ["juice", "soda", "drink", "water"]):
            return "beverage"
        else:
            return "generic"
    
    async def _generate_breakfast_spread_alternatives(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des alternatives pour les pâtes à tartiner."""
        await asyncio.sleep(0.02)
        
        alternatives = [
            {
                "id": "almond_butter",
                "name": "Pure Almond Butter",
                "category": "Nut Butter",
                "description": "100% almonds, no added sugar or palm oil",
                "nutrition_improvements": {
                    "sugar": "90% less sugar",
                    "protein": "2x more protein",
                    "fiber": "5x more fiber",
                    "vitamin_e": "Rich in vitamin E"
                },
                "health_benefits": [
                    "Heart-healthy monounsaturated fats",
                    "High protein content",
                    "No added sugars",
                    "Rich in vitamin E and magnesium"
                ],
                "estimated_score": 85,
                "availability": "Easy to find",
                "price_range": "$$"
            },
            {
                "id": "tahini",
                "name": "Sesame Tahini",
                "category": "Seed Butter",
                "description": "Ground sesame seeds, natural source of calcium",
                "nutrition_improvements": {
                    "sugar": "95% less sugar",
                    "calcium": "6x more calcium",
                    "protein": "Higher protein"
                },
                "health_benefits": [
                    "High in calcium",
                    "Good source of healthy fats",
                    "No added sugars",
                    "Rich in lignans"
                ],
                "estimated_score": 80,
                "availability": "Specialty stores",
                "price_range": "$$"
            }
        ]
        
        # Alternatives spécifiques aux contraintes santé
        if "diabetes" in (user_context.health_conditions or []):
            alternatives.append({
                "id": "sugar_free_spread",
                "name": "Sugar-Free Hazelnut Spread",
                "category": "Diabetic-Friendly",
                "description": "Hazelnut spread sweetened with stevia",
                "nutrition_improvements": {
                    "sugar": "100% less sugar",
                    "glycemic_index": "Very low"
                },
                "health_benefits": [
                    "Diabetic-friendly",
                    "No blood sugar spikes",
                    "Still satisfies hazelnut craving"
                ],
                "estimated_score": 75,
                "availability": "Health food stores",
                "price_range": "$$$"
            })
        
        return alternatives
    
    async def _generate_snack_alternatives(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des alternatives pour les snacks."""
        await asyncio.sleep(0.02)
        
        return [
            {
                "id": "mixed_nuts",
                "name": "Mixed Raw Nuts",
                "category": "Natural Snack",
                "description": "Unsalted mixed nuts (almonds, walnuts, hazelnuts)",
                "nutrition_improvements": {
                    "protein": "3x more protein",
                    "fiber": "4x more fiber",
                    "sodium": "95% less sodium",
                    "omega3": "Rich in omega-3"
                },
                "health_benefits": [
                    "Heart-healthy fats",
                    "High protein and fiber",
                    "No added sugars or salt",
                    "Sustained energy"
                ],
                "estimated_score": 90,
                "availability": "Everywhere",
                "price_range": "$$"
            }
        ]
    
    async def _generate_beverage_alternatives(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des alternatives pour les boissons."""
        await asyncio.sleep(0.02)
        
        return [
            {
                "id": "herbal_tea",
                "name": "Herbal Tea (unsweetened)",
                "category": "Natural Beverage",
                "description": "Various herbal teas with natural flavors",
                "nutrition_improvements": {
                    "calories": "100% less calories",
                    "sugar": "100% less sugar",
                    "antioxidants": "Rich in antioxidants"
                },
                "health_benefits": [
                    "Zero calories",
                    "Hydrating",
                    "Antioxidant properties",
                    "Various flavors available"
                ],
                "estimated_score": 95,
                "availability": "Everywhere",
                "price_range": "$"
            }
        ]
    
    async def _generate_generic_alternatives(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des alternatives génériques."""
        await asyncio.sleep(0.01)
        
        return [
            {
                "id": "whole_food_alternative",
                "name": "Whole Food Alternative",
                "category": "Natural",
                "description": "Less processed version of similar food",
                "nutrition_improvements": {
                    "processing": "Minimal processing",
                    "additives": "No artificial additives"
                },
                "health_benefits": [
                    "Less processed",
                    "More natural nutrients",
                    "No artificial additives"
                ],
                "estimated_score": 70,
                "availability": "Natural food stores",
                "price_range": "$$"
            }
        ]
    
    async def _generate_health_specific_alternatives(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des alternatives spécifiques aux conditions de santé."""
        await asyncio.sleep(0.02)
        
        alternatives = []
        
        for condition in (user_context.health_conditions or []):
            if condition == "hypertension":
                alternatives.append({
                    "id": "low_sodium_version",
                    "name": "Low-Sodium Version",
                    "category": "Heart-Healthy",
                    "description": "Same product type with reduced sodium",
                    "nutrition_improvements": {
                        "sodium": "75% less sodium"
                    },
                    "health_benefits": [
                        "Blood pressure friendly",
                        "Reduced cardiovascular risk"
                    ],
                    "estimated_score": 70,
                    "availability": "Health section",
                    "price_range": "$$"
                })
        
        return alternatives
    
    async def _prioritize_alternatives(
        self,
        alternatives: List[Dict[str, Any]],
        user_context: UserHealthContext,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Priorise les alternatives selon le profil utilisateur."""
        await asyncio.sleep(0.01)
        
        # Calcul du score de priorité pour chaque alternative
        for alternative in alternatives:
            priority_score = alternative.get("estimated_score", 50)
            
            # Bonus pour les contraintes santé
            for condition in (user_context.health_conditions or []):
                if condition in alternative.get("category", "").lower():
                    priority_score += 15
            
            # Bonus pour résoudre les problèmes identifiés
            for issue in issues:
                if issue["type"] in str(alternative.get("nutrition_improvements", {})):
                    priority_score += 10
            
            # Pénalité pour la disponibilité
            availability = alternative.get("availability", "")
            if "specialty" in availability.lower():
                priority_score -= 5
            
            alternative["priority_score"] = priority_score
        
        # Tri par score de priorité
        return sorted(alternatives, key=lambda x: x.get("priority_score", 0), reverse=True)


class ChatContextPreparationNode(BaseWorkflowNode):
    """
    Node de préparation du contexte pour le chat.
    
    Responsabilités :
    - Préparer un résumé conversationnel
    - Structurer les informations pour le LLM
    - Adapter le ton selon le profil utilisateur
    - Préparer les follow-up questions
    """
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.CHAT_CONTEXT_PREPARATION
    
    async def _execute_business_logic(self, state: WorkflowState) -> WorkflowState:
        """Prépare le contexte pour le chat."""
        # Génération du contexte conversationnel
        chat_context = await self._build_chat_context(
            state.nutrition_data,
            state.user_context,
            state.calculated_score,
            state.metadata.get("scores", {}),
            state.expert_analysis,
            state.alternatives,
            state.metadata.get("risk_assessment", {})
        )
        
        # Adaptation du ton
        adapted_context = await self._adapt_communication_style(
            chat_context,
            state.user_context,
            state.analysis_config
        )
        
        # Préparation des follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(
            state.nutrition_data,
            state.user_context,
            state.alternatives
        )
        
        return replace(
            state,
            chat_context={
                **adapted_context,
                "follow_up_questions": follow_up_questions,
                "preparation_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _build_chat_context(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        overall_score: float,
        scores: Dict[str, Any],
        expert_analysis: Optional[Dict[str, Any]],
        alternatives: Optional[List[Dict[str, Any]]],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construit le contexte pour le chat."""
        await asyncio.sleep(0.03)  # Simulation
        
        # Résumé du produit
        product_summary = self._create_product_summary(nutrition_data, overall_score)
        
        # Points clés à mentionner
        key_points = self._extract_key_points(nutrition_data, scores, expert_analysis, risk_assessment)
        
        # Recommandations prioritaires
        top_recommendations = self._prioritize_recommendations(alternatives, user_context)
        
        # Contexte de santé pertinent
        health_context = self._build_health_context(user_context, risk_assessment)
        
        return {
            "product_summary": product_summary,
            "key_points": key_points,
            "recommendations": top_recommendations,
            "health_context": health_context,
            "overall_assessment": self._create_overall_assessment(overall_score, scores),
            "conversational_hooks": self._create_conversational_hooks(nutrition_data, user_context)
        }
    
    def _create_product_summary(self, nutrition_data: NutritionData, score: float) -> Dict[str, Any]:
        """Crée un résumé conversationnel du produit."""
        score_category = "excellent" if score >= 80 else "good" if score >= 60 else "average" if score >= 40 else "poor"
        
        return {
            "name": nutrition_data.product_name or "This product",
            "score": score,
            "score_category": score_category,
            "brief_description": f"This product scores {score}/100, which is considered {score_category}.",
            "main_characteristics": self._identify_main_characteristics(nutrition_data)
        }
    
    def _identify_main_characteristics(self, nutrition_data: NutritionData) -> List[str]:
        """Identifie les caractéristiques principales du produit."""
        characteristics = []
        
        if nutrition_data.calories and nutrition_data.calories > 400:
            characteristics.append("high-calorie")
        elif nutrition_data.calories and nutrition_data.calories < 100:
            characteristics.append("low-calorie")
        
        if nutrition_data.sugar and nutrition_data.sugar > 15:
            characteristics.append("high-sugar")
        elif nutrition_data.sugar and nutrition_data.sugar < 5:
            characteristics.append("low-sugar")
        
        if nutrition_data.protein and nutrition_data.protein > 15:
            characteristics.append("protein-rich")
        
        if nutrition_data.fiber and nutrition_data.fiber > 6:
            characteristics.append("high-fiber")
        
        if nutrition_data.sodium and nutrition_data.sodium > 1:
            characteristics.append("high-sodium")
        
        return characteristics
    
    def _extract_key_points(
        self,
        nutrition_data: NutritionData,
        scores: Dict[str, Any],
        expert_analysis: Optional[Dict[str, Any]],
        risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extrait les points clés à mentionner."""
        key_points = []
        
        # Points basés sur le Nutri-Score
        nutri_score = scores.get("nutri_score", {})
        if nutri_score.get("letter") in ["A", "B"]:
            key_points.append({
                "type": "positive",
                "message": f"Good Nutri-Score rating ({nutri_score['letter']})"
            })
        elif nutri_score.get("letter") in ["D", "E"]:
            key_points.append({
                "type": "negative",
                "message": f"Poor Nutri-Score rating ({nutri_score['letter']})"
            })
        
        # Points basés sur les risques
        high_risks = [r for r in risk_assessment.get("risks", []) if r["severity"] == "high"]
        if high_risks:
            key_points.append({
                "type": "warning",
                "message": f"Contains allergens you should avoid"
            })
        
        # Points nutritionnels saillants
        if nutrition_data.fiber and nutrition_data.fiber > 10:
            key_points.append({
                "type": "positive", 
                "message": "Excellent source of fiber"
            })
        
        if nutrition_data.protein and nutrition_data.protein > 20:
            key_points.append({
                "type": "positive",
                "message": "High protein content"
            })
        
        return key_points
    
    def _prioritize_recommendations(
        self,
        alternatives: Optional[List[Dict[str, Any]]],
        user_context: UserHealthContext
    ) -> List[Dict[str, Any]]:
        """Priorise les recommandations."""
        if not alternatives:
            return []
        
        # Prendre les 3 meilleures alternatives
        top_alternatives = alternatives[:3]
        
        recommendations = []
        for alt in top_alternatives:
            recommendations.append({
                "name": alt["name"],
                "reason": self._generate_recommendation_reason(alt, user_context),
                "key_benefit": alt["health_benefits"][0] if alt.get("health_benefits") else "Healthier option"
            })
        
        return recommendations
    
    def _generate_recommendation_reason(
        self,
        alternative: Dict[str, Any],
        user_context: UserHealthContext
    ) -> str:
        """Génère la raison de recommandation."""
        improvements = alternative.get("nutrition_improvements", {})
        
        if "sugar" in improvements and "diabetes" in (user_context.health_conditions or []):
            return "Much lower sugar content, better for blood sugar control"
        elif "sodium" in improvements and "hypertension" in (user_context.health_conditions or []):
            return "Lower sodium content, better for blood pressure"
        elif "protein" in improvements:
            return "Higher protein content for better nutrition"
        else:
            return "Overall better nutritional profile"
    
    def _build_health_context(
        self,
        user_context: UserHealthContext,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construit le contexte santé pertinent."""
        return {
            "conditions": user_context.health_conditions or [],
            "allergies": user_context.allergies or [],
            "risks_identified": len(risk_assessment.get("risks", [])),
            "safe_for_user": risk_assessment.get("safe_for_user", True),
            "personalization_applied": len(user_context.health_conditions or []) > 0
        }
    
    def _create_overall_assessment(self, score: float, scores: Dict[str, Any]) -> str:
        """Crée une évaluation globale."""
        if score >= 80:
            return "This is a nutritionally excellent choice with great health benefits."
        elif score >= 60:
            return "This is a good choice with some room for improvement."
        elif score >= 40:
            return "This is an average choice - there are healthier alternatives available."
        else:
            return "This product has significant nutritional concerns - I'd recommend considering alternatives."
    
    def _create_conversational_hooks(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext
    ) -> List[str]:
        """Crée des accroches conversationnelles."""
        hooks = []
        
        if nutrition_data.product_name and "chocolate" in nutrition_data.product_name.lower():
            hooks.append("I see you're looking at a chocolate product!")
        
        if user_context.health_conditions:
            hooks.append("I've taken your health conditions into account in this analysis.")
        
        if nutrition_data.allergens:
            hooks.append("I noticed this product contains some allergens - let me break that down for you.")
        
        return hooks
    
    async def _adapt_communication_style(
        self,
        context: Dict[str, Any],
        user_context: UserHealthContext,
        config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Adapte le style de communication."""
        await asyncio.sleep(0.01)  # Simulation
        
        # Adaptation selon la qualité d'analyse demandée
        if config.quality_level.value == "expert":
            context["tone"] = "detailed_scientific"
            context["complexity"] = "high"
        elif config.quality_level.value == "premium":
            context["tone"] = "informative_friendly"
            context["complexity"] = "medium"
        else:
            context["tone"] = "simple_friendly"
            context["complexity"] = "low"
        
        # Adaptation selon l'âge (si disponible)
        if user_context.age and user_context.age < 25:
            context["style_notes"] = ["use_modern_language", "include_analogies"]
        elif user_context.age and user_context.age > 60:
            context["style_notes"] = ["use_clear_language", "focus_on_health_benefits"]
        
        return context
    
    async def _generate_follow_up_questions(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        alternatives: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Génère des questions de suivi."""
        await asyncio.sleep(0.01)  # Simulation
        
        questions = []
        
        # Questions basées sur les alternatives
        if alternatives and len(alternatives) > 0:
            questions.append("Would you like me to explain more about any of these healthier alternatives?")
            questions.append("Are you interested in where to find these recommended products?")
        
        # Questions basées sur les conditions de santé
        if user_context.health_conditions:
            questions.append("Would you like specific advice for managing your health condition through nutrition?")
        
        # Questions générales
        questions.extend([
            "Do you have any specific nutritional goals I can help you with?",
            "Would you like me to analyze any other products for comparison?",
            "Are there any ingredients you're particularly concerned about?"
        ])
        
        return questions[:3]  # Limiter à 3 questions


class ResponseAssemblyNode(BaseWorkflowNode):
    """
    Node d'assemblage de la réponse finale.
    
    Responsabilités :
    - Assembler toutes les données en réponse cohérente
    - Formater selon le format de sortie demandé
    - Valider la complétude de la réponse
    - Optimiser pour la performance
    """
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.RESPONSE_ASSEMBLY
    
    async def _execute_business_logic(self, state: WorkflowState) -> WorkflowState:
        """Assemble la réponse finale."""
        # Validation des données requises
        missing_data = self._validate_required_data(state)
        if missing_data:
            self._logger.warning(
                "Missing data for response assembly",
                missing_data=missing_data,
                workflow_id=state.workflow_id
            )
        
        # Assemblage de la réponse selon le format
        response_format = state.analysis_config.response_format
        
        if response_format == "detailed":
            assembled_response = await self._assemble_detailed_response(state)
        elif response_format == "summary":
            assembled_response = await self._assemble_summary_response(state)
        elif response_format == "chat_ready":
            assembled_response = await self._assemble_chat_ready_response(state)
        else:
            assembled_response = await self._assemble_standard_response(state)
        
        # Validation finale
        validation_results = await self._validate_response_completeness(assembled_response)
        
        return replace(
            state,
            metadata={
                **state.metadata,
                "assembled_response": assembled_response,
                "response_validation": validation_results,
                "assembly_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _validate_required_data(self, state: WorkflowState) -> List[str]:
        """Valide que toutes les données requises sont présentes."""
        missing = []
        
        if not state.nutrition_data:
            missing.append("nutrition_data")
        
        if state.calculated_score is None:
            missing.append("calculated_score")
        
        if not state.metadata.get("scores"):
            missing.append("detailed_scores")
        
        return missing
    
    async def _assemble_detailed_response(self, state: WorkflowState) -> Dict[str, Any]:
        """Assemble une réponse détaillée."""
        await asyncio.sleep(0.02)  # Simulation
        
        return {
            "analysis_id": state.workflow_id,
            "product": {
                "name": state.nutrition_data.product_name,
                "brand": state.nutrition_data.brand,
                "barcode": state.nutrition_data.barcode
            },
            "nutrition": self._format_nutrition_data(state.nutrition_data),
            "scores": {
                "overall": state.calculated_score,
                **state.metadata.get("scores", {})
            },
            "health_assessment": {
                "risks": state.metadata.get("risk_assessment", {}),
                "personalization": state.metadata.get("health_context", {}),
                "recommendations": state.alternatives[:5] if state.alternatives else []
            },
            "expert_analysis": state.expert_analysis,
            "alternatives": state.alternatives,
            "chat_context": state.chat_context,
            "metadata": {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "quality_level": state.analysis_config.quality_level.value,
                "processing_time_ms": self._calculate_processing_time(state)
            }
        }
    
    async def _assemble_summary_response(self, state: WorkflowState) -> Dict[str, Any]:
        """Assemble une réponse résumée."""
        await asyncio.sleep(0.01)  # Simulation
        
        # Top 3 points clés
        key_insights = self._extract_top_insights(state)
        
        return {
            "analysis_id": state.workflow_id,
            "product_name": state.nutrition_data.product_name,
            "overall_score": state.calculated_score,
            "nutri_score": state.metadata.get("scores", {}).get("nutri_score", {}).get("letter"),
            "key_insights": key_insights,
            "top_recommendations": state.alternatives[:3] if state.alternatives else [],
            "health_alerts": self._extract_health_alerts(state),
            "summary": self._generate_executive_summary(state)
        }
    
    async def _assemble_chat_ready_response(self, state: WorkflowState) -> Dict[str, Any]:
        """Assemble une réponse optimisée pour le chat."""
        await asyncio.sleep(0.01)  # Simulation
        
        return {
            "conversation_starter": self._generate_conversation_starter(state),
            "key_points": state.chat_context.get("key_points", [])[:3],
            "recommendations": state.chat_context.get("recommendations", [])[:2],
            "follow_up_questions": state.chat_context.get("follow_up_questions", []),
            "detailed_data": {
                "scores": state.metadata.get("scores", {}),
                "nutrition": self._format_nutrition_data(state.nutrition_data),
                "alternatives": state.alternatives
            }
        }
    
    async def _assemble_standard_response(self, state: WorkflowState) -> Dict[str, Any]:
        """Assemble une réponse standard."""
        await asyncio.sleep(0.01)  # Simulation
        
        return {
            "product": state.nutrition_data.product_name,
            "score": state.calculated_score,
            "nutrition": self._format_nutrition_data(state.nutrition_data),
            "recommendations": state.alternatives[:3] if state.alternatives else [],
            "health_notes": self._extract_health_alerts(state)
        }
    
    def _format_nutrition_data(self, nutrition_data: NutritionData) -> Dict[str, Any]:
        """Formate les données nutritionnelles."""
        return {
            "calories": nutrition_data.calories,
            "protein": nutrition_data.protein,
            "carbohydrates": nutrition_data.carbohydrates,
            "total_fat": nutrition_data.total_fat,
            "fiber": nutrition_data.fiber,
            "sugar": nutrition_data.sugar,
            "sodium": nutrition_data.sodium,
            "ingredients": nutrition_data.ingredients,
            "allergens": nutrition_data.allergens
        }
    
    def _extract_top_insights(self, state: WorkflowState) -> List[str]:
        """Extrait les 3 insights les plus importants."""
        insights = []
        
        # Insight sur le score
        score = state.calculated_score
        if score >= 80:
            insights.append(f"Excellent nutritional quality (score: {score}/100)")
        elif score <= 40:
            insights.append(f"Poor nutritional quality (score: {score}/100)")
        
        # Insight sur les risques
        risks = state.metadata.get("risk_assessment", {}).get("risks", [])
        high_risks = [r for r in risks if r["severity"] == "high"]
        if high_risks:
            insights.append(f"Contains {len(high_risks)} high-risk allergen(s)")
        
        # Insight sur les alternatives
        if state.alternatives and len(state.alternatives) > 0:
            best_alt = state.alternatives[0]
            insights.append(f"Better alternative available: {best_alt['name']}")
        
        return insights[:3]
    
    def _extract_health_alerts(self, state: WorkflowState) -> List[str]:
        """Extrait les alertes santé importantes."""
        alerts = []
        
        risks = state.metadata.get("risk_assessment", {}).get("risks", [])
        for risk in risks:
            if risk["severity"] in ["high", "medium"]:
                alerts.append(risk["message"])
        
        return alerts
    
    def _generate_executive_summary(self, state: WorkflowState) -> str:
        """Génère un résumé exécutif."""
        score = state.calculated_score
        product_name = state.nutrition_data.product_name or "This product"
        
        if score >= 80:
            return f"{product_name} is an excellent nutritional choice with high quality ingredients and balanced nutrition."
        elif score >= 60:
            return f"{product_name} is a good choice with some nutritional benefits, though alternatives may offer better value."
        elif score >= 40:
            return f"{product_name} has average nutritional quality - consider healthier alternatives for better nutrition."
        else:
            return f"{product_name} has significant nutritional concerns and healthier alternatives are strongly recommended."
    
    def _generate_conversation_starter(self, state: WorkflowState) -> str:
        """Génère une phrase d'accroche pour la conversation."""
        product_name = state.nutrition_data.product_name or "this product"
        score = state.calculated_score
        
        hooks = state.chat_context.get("conversational_hooks", [])
        if hooks:
            return hooks[0]
        
        if score >= 80:
            return f"Great choice! {product_name} is nutritionally excellent."
        elif score >= 60:
            return f"I've analyzed {product_name} - it's a decent choice with some good points."
        else:
            return f"I've found some concerns with {product_name} and have better alternatives to suggest."
    
    def _calculate_processing_time(self, state: WorkflowState) -> float:
        """Calcule le temps de traitement total."""
        if not state.processing_history:
            return 0.0
        
        start_time = state.processing_history[0].timestamp
        end_time = state.processing_history[-1].timestamp
        
        return (end_time - start_time).total_seconds() * 1000  # ms
    
    async def _validate_response_completeness(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Valide la complétude de la réponse."""
        await asyncio.sleep(0.005)  # Simulation
        
        validation = {
            "is_complete": True,
            "missing_fields": [],
            "warnings": []
        }
        
        # Vérifications de base
        required_fields = ["analysis_id", "product_name", "overall_score"] if "analysis_id" in response else ["product", "score"]
        
        for field in required_fields:
            if field not in response:
                validation["missing_fields"].append(field)
                validation["is_complete"] = False
        
        # Vérifications de qualité
        if isinstance(response.get("recommendations"), list) and len(response["recommendations"]) == 0:
            validation["warnings"].append("No recommendations generated")
        
        return validation