"""
Nodes restants du workflow d'analyse nutritionnelle.
Implements les stages finaux du pipeline d'analyse.

Architecture Pattern : Strategy + Template Method + Factory
Inspiration : MediatR.NET, NestJS Guards, Spring Boot Components
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import replace
from datetime import datetime, timedelta
import structlog

from app.workflows.interfaces import (
    IWorkflowNode, WorkflowState, WorkflowStage, WorkflowNodeError,
    InputData, AnalysisConfig
)
from app.workflows.nodes.core_nodes import BaseWorkflowNode
from app.models.clinical import NutritionData
from app.models.health import UserHealthContext

logger = structlog.get_logger(__name__)


class NutritionEnrichmentNode(BaseWorkflowNode):
    """
    Node d'enrichissement nutritionnel.
    
    Responsabilités :
    - Enrichir les données extraites avec des sources externes
    - Calculer les valeurs nutritionnelles dérivées
    - Normaliser les unités de mesure
    - Valider la cohérence nutritionnelle
    """
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.NUTRITION_ENRICHMENT
    
    async def _execute_business_logic(self, state: WorkflowState) -> WorkflowState:
        """Enrichit les données nutritionnelles."""
        if not state.nutrition_data:
            raise WorkflowNodeError(
                "No nutrition data to enrich",
                self.required_stage.value,
                recoverable=False
            )
        
        enriched_data = await self._enrich_nutrition_data(
            state.nutrition_data,
            state.analysis_config
        )
        
        # Validation de la cohérence
        validation_errors = await self._validate_nutrition_consistency(enriched_data)
        if validation_errors:
            self._logger.warning(
                "Nutrition data validation warnings",
                errors=validation_errors,
                workflow_id=state.workflow_id
            )
        
        return replace(
            state,
            nutrition_data=enriched_data,
            metadata={
                **state.metadata,
                "enrichment_sources": ["openfoodfacts", "usda", "ciqual"],
                "validation_warnings": validation_errors,
                "enrichment_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _enrich_nutrition_data(
        self, 
        base_data: NutritionData, 
        config: AnalysisConfig
    ) -> NutritionData:
        """
        Enrichit les données nutritionnelles.
        
        Strategy Pattern : Différentes sources d'enrichissement
        """
        # Simulation d'enrichissement (remplacer par vraies sources)
        await asyncio.sleep(0.1)  # Simulation API call
        
        # Calcul des valeurs dérivées
        derived_values = self._calculate_derived_nutrition_values(base_data)
        
        # Enrichissement par OpenFoodFacts (mock)
        if base_data.barcode:
            openfoodfacts_data = await self._enrich_from_openfoodfacts(base_data.barcode)
            if openfoodfacts_data:
                # Merge des données
                base_data = self._merge_nutrition_data(base_data, openfoodfacts_data)
        
        # Enrichissement par bases de données nutritionnelles
        if config.quality_level.value in ["premium", "expert"]:
            enhanced_data = await self._enrich_from_nutrition_databases(base_data)
            if enhanced_data:
                base_data = enhanced_data
        
        # Application des valeurs dérivées
        return replace(
            base_data,
            **derived_values,
            data_source=f"{base_data.data_source},enriched"
        )
    
    def _calculate_derived_nutrition_values(self, data: NutritionData) -> Dict[str, Any]:
        """Calcule les valeurs nutritionnelles dérivées."""
        derived = {}
        
        # Calcul des calories si manquantes
        if not data.calories and data.protein and data.carbohydrates and data.total_fat:
            # 4 kcal/g protéines, 4 kcal/g glucides, 9 kcal/g lipides
            calculated_calories = (
                (data.protein or 0) * 4 +
                (data.carbohydrates or 0) * 4 +
                (data.total_fat or 0) * 9
            )
            derived["calories"] = round(calculated_calories, 1)
        
        # Calcul du pourcentage de macronutriments
        total_macros = (data.protein or 0) + (data.carbohydrates or 0) + (data.total_fat or 0)
        if total_macros > 0:
            derived["protein_percentage"] = round((data.protein or 0) / total_macros * 100, 1)
            derived["carbs_percentage"] = round((data.carbohydrates or 0) / total_macros * 100, 1)
            derived["fat_percentage"] = round((data.total_fat or 0) / total_macros * 100, 1)
        
        # Densité nutritionnelle
        if data.calories and data.calories > 0:
            protein_density = (data.protein or 0) * 4 / data.calories
            derived["protein_density"] = round(protein_density, 3)
        
        return derived
    
    async def _enrich_from_openfoodfacts(self, barcode: str) -> Optional[Dict[str, Any]]:
        """Enrichit depuis OpenFoodFacts (mock implementation)."""
        await asyncio.sleep(0.05)  # Simulation API call
        
        # Mock data based on barcode
        mock_data = {
            "3017620422003": {  # Nutella
                "product_name": "Nutella Hazelnut Spread",
                "brand": "Ferrero",
                "additives": ["E322", "E476"],
                "nova_group": 4,
                "nutriscore": "E",
                "ecoscore": "D"
            }
        }
        
        return mock_data.get(barcode)
    
    async def _enrich_from_nutrition_databases(self, data: NutritionData) -> Optional[NutritionData]:
        """Enrichit depuis les bases de données nutritionnelles."""
        await asyncio.sleep(0.1)  # Simulation database query
        
        # Mock enrichment
        if "chocolate" in (data.product_name or "").lower():
            return replace(
                data,
                vitamins={"vitamin_e": 0.9, "vitamin_b1": 0.03},
                minerals={"magnesium": 64, "iron": 4.2},
                antioxidants={"flavonoids": 13.2}
            )
        
        return data
    
    def _merge_nutrition_data(self, base: NutritionData, additional: Dict[str, Any]) -> NutritionData:
        """Merge les données nutritionnelles."""
        merged_dict = base.__dict__.copy()
        
        # Merge intelligent des champs
        for key, value in additional.items():
            if key in merged_dict and merged_dict[key] is None:
                merged_dict[key] = value
            elif key not in merged_dict:
                merged_dict[key] = value
        
        return NutritionData(**merged_dict)
    
    async def _validate_nutrition_consistency(self, data: NutritionData) -> List[str]:
        """Valide la cohérence des données nutritionnelles."""
        warnings = []
        
        # Validation des calories vs macronutriments
        if data.calories and data.protein and data.carbohydrates and data.total_fat:
            calculated_calories = (
                data.protein * 4 + data.carbohydrates * 4 + data.total_fat * 9
            )
            
            calorie_diff = abs(data.calories - calculated_calories)
            if calorie_diff > data.calories * 0.15:  # Plus de 15% de différence
                warnings.append(
                    f"Calorie inconsistency: declared {data.calories}, "
                    f"calculated {calculated_calories:.1f}"
                )
        
        # Validation des valeurs aberrantes
        if data.protein and data.protein > 100:
            warnings.append(f"Protein content seems too high: {data.protein}g")
        
        if data.sodium and data.sodium > 10:  # Plus de 10g de sodium
            warnings.append(f"Sodium content seems too high: {data.sodium}g")
        
        return warnings


class HealthProfileContextNode(BaseWorkflowNode):
    """
    Node de contextualisation du profil santé.
    
    Responsabilités :
    - Analyser le profil santé de l'utilisateur
    - Identifier les contraintes alimentaires
    - Calculer les besoins nutritionnels personnalisés
    - Évaluer les risques potentiels
    """
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.HEALTH_PROFILE_CONTEXT
    
    async def _execute_business_logic(self, state: WorkflowState) -> WorkflowState:
        """Contextualise avec le profil santé."""
        health_context = await self._analyze_health_profile(
            state.user_context,
            state.nutrition_data
        )
        
        personalized_needs = await self._calculate_personalized_needs(
            state.user_context
        )
        
        risk_assessment = await self._assess_health_risks(
            state.user_context,
            state.nutrition_data
        )
        
        return replace(
            state,
            metadata={
                **state.metadata,
                "health_context": health_context,
                "personalized_needs": personalized_needs,
                "risk_assessment": risk_assessment,
                "health_analysis_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _analyze_health_profile(
        self,
        user_context: UserHealthContext,
        nutrition_data: NutritionData
    ) -> Dict[str, Any]:
        """Analyse le profil santé."""
        await asyncio.sleep(0.05)  # Simulation
        
        context = {
            "age_category": self._categorize_age(user_context.age),
            "bmi_category": self._categorize_bmi(user_context.weight, user_context.height),
            "activity_level": user_context.activity_level.value if user_context.activity_level else "moderate",
            "dietary_restrictions": user_context.dietary_restrictions or [],
            "health_conditions": user_context.health_conditions or [],
            "allergies": user_context.allergies or []
        }
        
        # Analyse des contraintes spécifiques
        constraints = []
        
        if "diabetes" in context["health_conditions"]:
            constraints.append({
                "type": "diabetes",
                "recommendations": ["low_sugar", "complex_carbs", "fiber_rich"],
                "warnings": ["high_sugar", "simple_carbs"]
            })
        
        if "hypertension" in context["health_conditions"]:
            constraints.append({
                "type": "hypertension",
                "recommendations": ["low_sodium", "potassium_rich"],
                "warnings": ["high_sodium", "processed_foods"]
            })
        
        context["health_constraints"] = constraints
        
        return context
    
    def _categorize_age(self, age: Optional[int]) -> str:
        """Catégorise l'âge."""
        if not age:
            return "adult"
        
        if age < 18:
            return "minor"
        elif age < 65:
            return "adult"
        else:
            return "senior"
    
    def _categorize_bmi(self, weight: Optional[float], height: Optional[float]) -> str:
        """Catégorise l'IMC."""
        if not weight or not height:
            return "unknown"
        
        bmi = weight / ((height / 100) ** 2)
        
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"
    
    async def _calculate_personalized_needs(
        self,
        user_context: UserHealthContext
    ) -> Dict[str, Any]:
        """Calcule les besoins nutritionnels personnalisés."""
        await asyncio.sleep(0.03)  # Simulation
        
        # Calcul des besoins caloriques (formule Harris-Benedict simplifiée)
        if user_context.age and user_context.weight and user_context.height:
            if user_context.gender == "male":
                bmr = 88.362 + (13.397 * user_context.weight) + (4.799 * user_context.height) - (5.677 * user_context.age)
            else:
                bmr = 447.593 + (9.247 * user_context.weight) + (3.098 * user_context.height) - (4.330 * user_context.age)
            
            # Facteur d'activité
            activity_factors = {
                "sedentary": 1.2,
                "light": 1.375,
                "moderate": 1.55,
                "active": 1.725,
                "very_active": 1.9
            }
            
            activity_level = user_context.activity_level.value if user_context.activity_level else "moderate"
            daily_calories = bmr * activity_factors.get(activity_level, 1.55)
            
        else:
            daily_calories = 2000  # Valeur par défaut
        
        return {
            "daily_calories": round(daily_calories),
            "protein_g": round(daily_calories * 0.15 / 4),  # 15% des calories
            "carbs_g": round(daily_calories * 0.50 / 4),    # 50% des calories
            "fat_g": round(daily_calories * 0.35 / 9),      # 35% des calories
            "fiber_g": 25,  # Recommandation générale
            "sodium_mg": 2300  # Limite recommandée
        }
    
    async def _assess_health_risks(
        self,
        user_context: UserHealthContext,
        nutrition_data: NutritionData
    ) -> Dict[str, Any]:
        """Évalue les risques santé."""
        await asyncio.sleep(0.02)  # Simulation
        
        risks = []
        
        # Analyse des allergènes
        if user_context.allergies and nutrition_data.allergens:
            common_allergens = set(user_context.allergies) & set(nutrition_data.allergens)
            if common_allergens:
                risks.append({
                    "type": "allergen_risk",
                    "severity": "high",
                    "allergens": list(common_allergens),
                    "message": f"Product contains allergens: {', '.join(common_allergens)}"
                })
        
        # Analyse sodium pour hypertension
        if "hypertension" in (user_context.health_conditions or []):
            if nutrition_data.sodium and nutrition_data.sodium > 1.5:  # > 1.5g
                risks.append({
                    "type": "sodium_risk",
                    "severity": "medium",
                    "value": nutrition_data.sodium,
                    "message": "High sodium content may affect blood pressure"
                })
        
        # Analyse sucre pour diabète
        if "diabetes" in (user_context.health_conditions or []):
            if nutrition_data.sugar and nutrition_data.sugar > 15:  # > 15g
                risks.append({
                    "type": "sugar_risk",
                    "severity": "medium",
                    "value": nutrition_data.sugar,
                    "message": "High sugar content may affect blood glucose"
                })
        
        return {
            "risks": risks,
            "risk_score": len([r for r in risks if r["severity"] == "high"]) * 2 + 
                         len([r for r in risks if r["severity"] == "medium"]),
            "safe_for_user": len(risks) == 0
        }


class ScoreCalculationNode(BaseWorkflowNode):
    """
    Node de calcul de score nutritionnel.
    
    Responsabilités :
    - Calculer le score Nutri-Score
    - Calculer des scores personnalisés selon le profil
    - Évaluer la qualité nutritionnelle globale
    - Fournir des recommandations quantifiées
    """
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.SCORE_CALCULATION
    
    async def _execute_business_logic(self, state: WorkflowState) -> WorkflowState:
        """Calcule les scores nutritionnels."""
        if not state.nutrition_data:
            raise WorkflowNodeError(
                "No nutrition data for score calculation",
                self.required_stage.value,
                recoverable=False
            )
        
        # Calcul du Nutri-Score
        nutri_score = await self._calculate_nutri_score(state.nutrition_data)
        
        # Calcul du score personnalisé
        personal_score = await self._calculate_personal_score(
            state.nutrition_data,
            state.user_context,
            state.metadata.get("health_context", {}),
            state.metadata.get("risk_assessment", {})
        )
        
        # Score de qualité globale
        quality_score = await self._calculate_quality_score(state.nutrition_data)
        
        # Score global combiné
        overall_score = self._combine_scores(nutri_score, personal_score, quality_score)
        
        return replace(
            state,
            calculated_score=overall_score,
            metadata={
                **state.metadata,
                "scores": {
                    "nutri_score": nutri_score,
                    "personal_score": personal_score,
                    "quality_score": quality_score,
                    "overall_score": overall_score
                },
                "score_calculation_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _calculate_nutri_score(self, nutrition_data: NutritionData) -> Dict[str, Any]:
        """Calcule le Nutri-Score officiel."""
        await asyncio.sleep(0.02)  # Simulation
        
        # Points négatifs (par 100g)
        negative_points = 0
        
        # Énergie (kJ)
        calories = nutrition_data.calories or 0
        energy_kj = calories * 4.184
        if energy_kj > 3350:
            negative_points += 10
        elif energy_kj > 3015:
            negative_points += 9
        # ... (table complète du Nutri-Score)
        
        # Sucres
        sugar = nutrition_data.sugar or 0
        if sugar > 45:
            negative_points += 10
        elif sugar > 40:
            negative_points += 9
        # ... (table complète)
        
        # Acides gras saturés (approximation)
        saturated_fat = (nutrition_data.total_fat or 0) * 0.3  # Estimation
        if saturated_fat > 10:
            negative_points += 10
        # ... (table complète)
        
        # Sodium
        sodium = nutrition_data.sodium or 0
        sodium_mg = sodium * 1000
        if sodium_mg > 900:
            negative_points += 10
        # ... (table complète)
        
        # Points positifs
        positive_points = 0
        
        # Fibres
        fiber = nutrition_data.fiber or 0
        if fiber > 4.7:
            positive_points += 5
        elif fiber > 3.7:
            positive_points += 4
        # ... (table complète)
        
        # Protéines
        protein = nutrition_data.protein or 0
        if protein > 8:
            positive_points += 5
        elif protein > 6.4:
            positive_points += 4
        # ... (table complète)
        
        # Fruits et légumes (approximation basée sur les ingrédients)
        fruits_veggies_points = self._estimate_fruits_vegetables_points(nutrition_data.ingredients)
        positive_points += fruits_veggies_points
        
        # Calcul final
        final_score = negative_points - positive_points
        
        # Conversion en lettre
        if final_score < -1:
            letter = "A"
        elif final_score < 3:
            letter = "B"
        elif final_score < 11:
            letter = "C"
        elif final_score < 19:
            letter = "D"
        else:
            letter = "E"
        
        return {
            "score": final_score,
            "letter": letter,
            "negative_points": negative_points,
            "positive_points": positive_points,
            "details": {
                "energy": energy_kj,
                "sugar": sugar,
                "saturated_fat": saturated_fat,
                "sodium": sodium_mg,
                "fiber": fiber,
                "protein": protein,
                "fruits_vegetables": fruits_veggies_points
            }
        }
    
    def _estimate_fruits_vegetables_points(self, ingredients: Optional[List[str]]) -> int:
        """Estime les points fruits/légumes basés sur les ingrédients."""
        if not ingredients:
            return 0
        
        # Mots-clés pour fruits et légumes
        fruit_veg_keywords = [
            "apple", "banana", "orange", "tomato", "carrot", "spinach",
            "pomme", "banane", "orange", "tomate", "carotte", "épinard",
            "fruit", "légume", "vegetable"
        ]
        
        ingredient_text = " ".join(ingredients).lower()
        
        for keyword in fruit_veg_keywords:
            if keyword in ingredient_text:
                return 2  # Points modérés
        
        return 0
    
    async def _calculate_personal_score(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        health_context: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcule un score personnalisé selon le profil utilisateur."""
        await asyncio.sleep(0.03)  # Simulation
        
        base_score = 50  # Score de base
        adjustments = []
        
        # Ajustements basés sur les conditions de santé
        for condition in (user_context.health_conditions or []):
            if condition == "diabetes":
                sugar_penalty = min((nutrition_data.sugar or 0) * 2, 20)
                base_score -= sugar_penalty
                adjustments.append(f"Diabetes: -{sugar_penalty} (sugar content)")
            
            elif condition == "hypertension":
                sodium_penalty = min((nutrition_data.sodium or 0) * 10, 15)
                base_score -= sodium_penalty
                adjustments.append(f"Hypertension: -{sodium_penalty} (sodium content)")
        
        # Bonus pour les besoins nutritionnels
        protein_bonus = min((nutrition_data.protein or 0) * 0.5, 10)
        base_score += protein_bonus
        adjustments.append(f"Protein bonus: +{protein_bonus}")
        
        fiber_bonus = min((nutrition_data.fiber or 0) * 2, 15)
        base_score += fiber_bonus
        adjustments.append(f"Fiber bonus: +{fiber_bonus}")
        
        # Pénalités pour les risques
        risk_penalty = risk_assessment.get("risk_score", 0) * 5
        base_score -= risk_penalty
        if risk_penalty > 0:
            adjustments.append(f"Health risks: -{risk_penalty}")
        
        # Normalisation (0-100)
        final_score = max(0, min(100, base_score))
        
        return {
            "score": round(final_score, 1),
            "base_score": 50,
            "adjustments": adjustments,
            "explanation": "Personalized score based on health profile and dietary needs"
        }
    
    async def _calculate_quality_score(self, nutrition_data: NutritionData) -> Dict[str, Any]:
        """Calcule un score de qualité nutritionnelle."""
        await asyncio.sleep(0.02)  # Simulation
        
        quality_factors = {}
        total_score = 0
        max_score = 0
        
        # Densité protéique
        if nutrition_data.calories and nutrition_data.calories > 0:
            protein_density = (nutrition_data.protein or 0) * 4 / nutrition_data.calories
            protein_score = min(protein_density * 50, 20)
            quality_factors["protein_density"] = protein_score
            total_score += protein_score
            max_score += 20
        
        # Rapport fibres/calories
        if nutrition_data.calories and nutrition_data.calories > 0:
            fiber_ratio = (nutrition_data.fiber or 0) / (nutrition_data.calories / 100)
            fiber_score = min(fiber_ratio * 10, 20)
            quality_factors["fiber_ratio"] = fiber_score
            total_score += fiber_score
            max_score += 20
        
        # Équilibre des macronutriments
        total_macros = (nutrition_data.protein or 0) + (nutrition_data.carbohydrates or 0) + (nutrition_data.total_fat or 0)
        if total_macros > 0:
            protein_pct = (nutrition_data.protein or 0) / total_macros * 100
            # Score optimal entre 15-25% de protéines
            if 15 <= protein_pct <= 25:
                balance_score = 15
            else:
                balance_score = max(0, 15 - abs(protein_pct - 20) * 0.5)
            quality_factors["macro_balance"] = balance_score
            total_score += balance_score
            max_score += 15
        
        # Pénalités pour additifs (estimation)
        if nutrition_data.ingredients:
            additive_penalty = len([ing for ing in nutrition_data.ingredients if ing.startswith("E")]) * 2
            quality_factors["additive_penalty"] = -additive_penalty
            total_score -= additive_penalty
        
        # Score final (0-100)
        if max_score > 0:
            final_score = (total_score / max_score) * 100
        else:
            final_score = 50
        
        final_score = max(0, min(100, final_score))
        
        return {
            "score": round(final_score, 1),
            "factors": quality_factors,
            "explanation": "Overall nutritional quality based on nutrient density and composition"
        }
    
    def _combine_scores(
        self,
        nutri_score: Dict[str, Any],
        personal_score: Dict[str, Any],
        quality_score: Dict[str, Any]
    ) -> float:
        """Combine les différents scores en un score global."""
        # Pondération des scores
        weights = {
            "nutri": 0.3,
            "personal": 0.5,
            "quality": 0.2
        }
        
        # Conversion du Nutri-Score en échelle 0-100
        nutri_numeric = {
            "A": 90, "B": 75, "C": 50, "D": 25, "E": 10
        }.get(nutri_score["letter"], 50)
        
        combined = (
            nutri_numeric * weights["nutri"] +
            personal_score["score"] * weights["personal"] +
            quality_score["score"] * weights["quality"]
        )
        
        return round(combined, 1)