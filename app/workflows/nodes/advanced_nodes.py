"""
Nodes avancés du workflow d'analyse nutritionnelle.
Implements l'extraction de données et l'analyse expert avec enrichissement LLM.

Architecture Pattern : Strategy + Factory + Command + DI
Inspiration : Chain of Responsibility, MediatR, Command Pattern
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import replace
from datetime import datetime
import structlog

from app.workflows.interfaces import (
    IWorkflowNode, WorkflowState, WorkflowStage, WorkflowNodeError,
    InputData, InputType, AnalysisConfig, QualityLevel
)
from app.workflows.nodes.core_nodes import BaseWorkflowNode
from app.workflows.container import workflow_container
from app.models.health import NutritionData, UserHealthContext
from app.services.ocr import IOCRService, IOCRProcessor, ocr_manager
from app.services.barcode import lookup_barcode, barcode_manager
from app.services.llm.interfaces import ILLMEnricher, LLMResult

logger = structlog.get_logger(__name__)


# ================================
# DATA EXTRACTION NODE
# ================================

class DataExtractionNode(BaseWorkflowNode):
    """
    Node d'extraction de données avec stratégies multiples et enrichissement LLM.
    
    Responsabilités :
    - Extraction OCR depuis images (service réel)
    - Lookup de codes-barres dans bases de données
    - Parsing de données JSON structurées
    - Enrichissement LLM pour améliorer qualité des données
    - Fallback automatique entre stratégies
    """
    
    def __init__(self, 
                 ocr_service: IOCRService = None, 
                 ocr_processor: IOCRProcessor = None,
                 llm_enricher: ILLMEnricher = None):
        super().__init__()
        self.ocr_service = ocr_service or ocr_manager.get_ocr_service()
        self.ocr_processor = ocr_processor or ocr_manager.get_nutrition_processor()
        
        # Injection de dépendance pour le service LLM
        self.llm_enricher = llm_enricher or self._resolve_llm_service()
    
    def _resolve_llm_service(self) -> Optional[ILLMEnricher]:
        """Résout le service LLM via le container DI."""
        try:
            return workflow_container.resolve(ILLMEnricher)
        except Exception as e:
            self._logger.warning("LLM service not available, continuing without enrichment", error=str(e))
            return None
    
    @property
    def node_name(self) -> str:
        return "data_extraction"
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.DATA_EXTRACTION
    
    async def can_process(self, state: WorkflowState) -> bool:
        """Peut traiter si tokens validés et réservés."""
        return (state.current_stage == WorkflowStage.TOKEN_VALIDATION and
                state.input_data is not None and
                "token_reservation" in state.input_data.complexity_hints)
    
    async def validate_preconditions(self, state: WorkflowState) -> List[str]:
        """Valide que nous avons une réservation de tokens valide."""
        errors = []
        
        token_reservation = state.input_data.complexity_hints.get("token_reservation")
        if not token_reservation or not token_reservation.get("success"):
            errors.append("Valid token reservation required")
        
        return errors
    
    async def _process_internal(self, state: WorkflowState) -> WorkflowState:
        """Exécute l'extraction selon le type d'input avec enrichissement LLM."""
        input_data = state.input_data
        
        try:
            # Sélection de la stratégie d'extraction
            if input_data.type == InputType.IMAGE:
                nutrition_data = await self._extract_from_image(input_data)
            elif input_data.type == InputType.BARCODE:
                nutrition_data = await self._extract_from_barcode(input_data)
            elif input_data.type == InputType.JSON_DATA:
                nutrition_data = await self._extract_from_json(input_data)
            else:
                raise WorkflowNodeError(
                    f"Unsupported input type: {input_data.type}",
                    self.required_stage.value,
                    recoverable=False
                )
            
            if not nutrition_data:
                # Tentative de fallback
                self._logger.warning("Primary extraction failed, trying fallback")
                nutrition_data = await self._fallback_extraction(input_data)
            
            if not nutrition_data:
                raise WorkflowNodeError(
                    "All extraction strategies failed",
                    self.required_stage.value,
                    recoverable=False
                )
            
            # Enrichissement LLM des données extraites
            enriched_nutrition_data, llm_metadata = await self._enrich_with_llm(nutrition_data, input_data)
            
            # Ajout des métadonnées LLM dans l'historique de traitement
            llm_step = {
                "stage": "llm_enrichment",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": llm_metadata
            }
            
            return replace(
                state,
                nutrition_data=enriched_nutrition_data,
                processing_history=state.processing_history + (llm_step,)
            )
            
        except Exception as e:
            self._logger.error("Data extraction failed", error=str(e))
            raise WorkflowNodeError(
                f"Data extraction failed: {str(e)}",
                self.required_stage.value,
                recoverable=True
            )
    
    def _get_next_stage(self) -> Optional[WorkflowStage]:
        return WorkflowStage.NUTRITION_ENRICHMENT
    
    async def _extract_from_image(self, input_data: InputData) -> Optional[NutritionData]:
        """Extraction depuis image avec OCR service réel."""
        if not input_data.image_data:
            return None
        
        try:
            # Validation de la qualité d'image
            quality_check = await self.ocr_service.validate_image_quality(input_data.image_data)
            
            if not quality_check.get("is_valid", False):
                self._logger.warning("Image quality check failed", warnings=quality_check.get("warnings", []))
                # Continue anyway, might still work
            
            # Extraction OCR
            ocr_result = await self.ocr_service.extract_text(input_data.image_data)
            
            if not ocr_result.raw_text.strip():
                self._logger.warning("No text extracted from image")
                return None
            
            # Post-traitement avec processeur nutrition
            enhanced_result = await self.ocr_processor.enhance_nutrition_extraction(ocr_result)
            
            # Conversion en NutritionData
            nutrition_data = await self._convert_ocr_to_nutrition_data(enhanced_result)
            
            self._logger.info(
                "Image extraction completed",
                confidence=ocr_result.confidence,
                text_length=len(ocr_result.raw_text),
                keywords_found=len(enhanced_result.nutrition_keywords)
            )
            
            return nutrition_data
            
        except Exception as e:
            self._logger.error("Image extraction failed", error=str(e))
            return None
    
    
    async def _extract_from_barcode(self, input_data: InputData) -> Optional[NutritionData]:
        """Extraction depuis code-barres avec service OpenFoodFacts réel."""
        if not input_data.barcode:
            return None
        
        try:
            self._logger.info(f"Looking up barcode via OpenFoodFacts: {input_data.barcode}")
            
            # Utilisation du service barcode réel
            barcode_result = await lookup_barcode(input_data.barcode)
            
            if not barcode_result:
                self._logger.warning(f"No product found for barcode: {input_data.barcode}")
                return None
            
            # Conversion BarcodeResult vers NutritionData
            nutrition_data = await self._convert_barcode_to_nutrition_data(barcode_result)
            
            self._logger.info(
                "Barcode lookup completed",
                barcode=input_data.barcode,
                product_name=nutrition_data.product_name,
                provider=barcode_result.provider,
                confidence=barcode_result.confidence
            )
            
            return nutrition_data
            
        except Exception as e:
            self._logger.error("Barcode lookup failed", barcode=input_data.barcode, error=str(e))
            return None
    
    async def _extract_from_json(self, input_data: InputData) -> Optional[NutritionData]:
        """Extraction depuis données JSON structurées."""
        if not input_data.json_data:
            return None
        
        try:
            json_data = input_data.json_data
            
            # Conversion directe JSON vers NutritionData
            nutrition_data = NutritionData(
                product_name=json_data.get("product_name", "Unknown Product"),
                brand=json_data.get("brand"),
                barcode=json_data.get("barcode"),
                serving_size=json_data.get("serving_size", "1 serving"),  # Valeur par défaut
                calories=json_data.get("calories", 0),
                protein=json_data.get("protein", 0),
                carbohydrates=json_data.get("carbohydrates", 0),
                total_fat=json_data.get("total_fat", 0),
                saturated_fat=json_data.get("saturated_fat"),
                trans_fat=json_data.get("trans_fat"),
                fiber=json_data.get("fiber"),
                sugar=json_data.get("sugar"),
                sodium=json_data.get("sodium"),
                potassium=json_data.get("potassium"),
                cholesterol=json_data.get("cholesterol"),
                ingredients=json_data.get("ingredients", []),  # Liste vide par défaut
                allergens=json_data.get("allergens", []),  # Liste vide par défaut
                data_source="json_input"
            )
            
            self._logger.info("JSON extraction completed")
            return nutrition_data
            
        except Exception as e:
            self._logger.error("JSON extraction failed", error=str(e))
            return None
    
    async def _fallback_extraction(self, input_data: InputData) -> Optional[NutritionData]:
        """Stratégie de fallback en cas d'échec des extractions principales."""
        self._logger.info("Attempting fallback extraction")
        
        # Fallback basique : données minimales
        nutrition_data = NutritionData(
            product_name="Unknown Product",
            brand=None,
            barcode=input_data.barcode if hasattr(input_data, 'barcode') else None,
            serving_size="1 serving",
            calories=0,
            protein=0.0,
            carbohydrates=0.0,
            total_fat=0.0,
            data_source="fallback"
        )
        
        return nutrition_data
    
    async def _convert_ocr_to_nutrition_data(self, ocr_result) -> NutritionData:
        """Convertit le résultat OCR en données nutritionnelles structurées."""
        # Extraction des valeurs nutritionnelles depuis le texte OCR
        nutrition_values = {}
        
        # Recherche des macronutriments dans les valeurs détectées
        for number_data in ocr_result.detected_numbers:
            measurement_type = number_data.get("type", "unknown")
            value = number_data.get("value", 0)
            unit = number_data.get("unit", "")
            
            # Normalisation des unités (conversion en g pour 100g)
            normalized_value = self._normalize_nutrition_value(value, unit, measurement_type)
            
            if measurement_type in ["calories", "protein", "fat", "carbohydrates", "fiber", "sugar", "sodium"]:
                nutrition_values[measurement_type] = normalized_value
        
        # Extraction du nom du produit depuis les premiers mots du texte
        text_lines = ocr_result.raw_text.split('\n')
        product_name = None
        for line in text_lines[:3]:  # Chercher dans les 3 premières lignes
            line = line.strip()
            if len(line) > 3 and not any(keyword in line.lower() for keyword in ["nutrition", "facts", "ingredients"]):
                product_name = line
                break
        
        # Extraction des ingrédients
        ingredients = []
        if ocr_result.ingredient_sections:
            # Parser la première section d'ingrédients trouvée
            ingredient_text = ocr_result.ingredient_sections[0]
            # Simplifier : split par virgules
            ingredients = [ing.strip() for ing in ingredient_text.split(',') if ing.strip()]
        
        # Détection des allergènes
        allergens = []
        text_lower = ocr_result.raw_text.lower()
        common_allergens = ["milk", "eggs", "fish", "nuts", "wheat", "soy", "gluten"]
        for allergen in common_allergens:
            if allergen in text_lower:
                allergens.append(allergen)
        
        return NutritionData(
            product_name=product_name or "Unknown Product",
            brand=None,  # Difficile à extraire automatiquement
            barcode=None,
            serving_size="100g",  # Portion standard
            calories=nutrition_values.get("calories", 0),
            protein=nutrition_values.get("protein", 0),
            carbohydrates=nutrition_values.get("carbohydrates", 0),
            total_fat=nutrition_values.get("fat", 0),
            fiber=nutrition_values.get("fiber"),
            sugar=nutrition_values.get("sugar"),
            sodium=nutrition_values.get("sodium"),
            ingredients=ingredients if ingredients else [],
            allergens=allergens if allergens else [],
            data_source="ocr_extraction"
        )
    
    def _normalize_nutrition_value(self, value: float, unit: str, measurement_type: str) -> Optional[float]:
        """Normalise les valeurs nutritionnelles (conversion en unités standard)."""
        if not value:
            return None
        
        unit_lower = unit.lower()
        
        # Conversion des calories
        if measurement_type == "calories":
            if unit_lower in ["kcal", "cal"]:
                return value
            elif unit_lower == "kj":
                return value / 4.184  # Conversion kJ vers kcal
        
        # Conversion des masses (vers grammes)
        elif measurement_type in ["protein", "fat", "carbohydrates", "fiber", "sugar"]:
            if unit_lower in ["g", "gram", "grams"]:
                return value
            elif unit_lower in ["mg", "milligram", "milligrams"]:
                return value / 1000  # mg vers g
            elif unit_lower in ["kg", "kilogram", "kilograms"]:
                return value * 1000  # kg vers g
        
        # Sodium (conversion vers grammes)
        elif measurement_type == "sodium":
            if unit_lower in ["mg", "milligram", "milligrams"]:
                return value / 1000  # mg vers g
            elif unit_lower in ["g", "gram", "grams"]:
                return value
        
        # Pourcentages (garder tel quel)
        elif unit_lower in ["%", "percent"]:
            return value
        
        # Valeur par défaut
        return value
    
    async def _convert_barcode_to_nutrition_data(self, barcode_result) -> NutritionData:
        """Convertit un BarcodeResult en NutritionData pour le workflow."""
        from app.services.barcode import BarcodeResult
        
        # Extraction des valeurs nutritionnelles (déjà normalisées)
        nutrition_facts = barcode_result.nutrition_facts or {}
        
        # Mapping direct des valeurs principales
        nutrition_data = NutritionData(
            product_name=barcode_result.product_name or "Unknown Product",
            brand=barcode_result.brand,
            barcode=barcode_result.barcode,
            serving_size=barcode_result.serving_size or "100g",
            
            # Macronutriments (directement depuis nutrition_facts normalisées)
            calories=nutrition_facts.get("calories", 0),
            protein=nutrition_facts.get("protein", 0),
            carbohydrates=nutrition_facts.get("carbohydrates", 0),
            total_fat=nutrition_facts.get("total_fat", 0),
            saturated_fat=nutrition_facts.get("saturated_fat"),
            trans_fat=nutrition_facts.get("trans_fat"),
            fiber=nutrition_facts.get("fiber"),
            sugar=nutrition_facts.get("sugar"),
            
            # Micronutriments
            sodium=nutrition_facts.get("sodium"),
            potassium=nutrition_facts.get("potassium"),
            cholesterol=nutrition_facts.get("cholesterol"),
            
            # Informations additionnelles
            ingredients=barcode_result.ingredients or [],
            allergens=barcode_result.allergens or [],
            
            # Métadonnées
            data_source=f"barcode_{barcode_result.provider.lower()}",
            confidence_score=barcode_result.confidence
        )
        
        self._logger.info(
            "Converted barcode result to nutrition data",
            barcode=barcode_result.barcode,
            provider=barcode_result.provider,
            data_quality=barcode_result.data_quality,
            nutrition_completeness=len([v for v in nutrition_facts.values() if v is not None])
        )
        
        return nutrition_data
    
    async def _enrich_with_llm(self, nutrition_data: NutritionData, input_data: InputData) -> tuple[NutritionData, Dict[str, Any]]:
        """
        Enrichit les données nutritionnelles avec l'analyse LLM.
        
        Args:
            nutrition_data: Données nutritionnelles de base
            input_data: Données d'entrée originales pour contexte
            
        Returns:
            tuple[NutritionData, Dict[str, Any]]: Données enrichies + métadonnées LLM
        """
        # Si pas de service LLM disponible, retourner données originales
        if not self.llm_enricher:
            self._logger.info("LLM enrichment skipped - service not available")
            return nutrition_data, {"status": "skipped", "reason": "service_unavailable"}
        
        try:
            self._logger.info("Starting LLM enrichment", product_name=nutrition_data.product_name)
            
            # Préparation des données pour enrichissement
            product_data = {
                "product_name": nutrition_data.product_name,
                "brand": nutrition_data.brand,
                "barcode": nutrition_data.barcode,
                "serving_size": nutrition_data.serving_size,
                "calories": nutrition_data.calories,
                "protein": nutrition_data.protein,
                "carbohydrates": nutrition_data.carbohydrates,
                "total_fat": nutrition_data.total_fat,
                "saturated_fat": nutrition_data.saturated_fat,
                "trans_fat": nutrition_data.trans_fat,
                "fiber": nutrition_data.fiber,
                "sugar": nutrition_data.sugar,
                "sodium": nutrition_data.sodium,
                "potassium": nutrition_data.potassium,
                "cholesterol": nutrition_data.cholesterol,
                "ingredients": nutrition_data.ingredients or [],
                "allergens": nutrition_data.allergens or [],
                "data_source": nutrition_data.data_source,
                "input_type": input_data.type.value if input_data else "unknown"
            }
            
            # Enrichissement via le service LLM
            enrichment_result: LLMResult = await self.llm_enricher.enrich_product_data(product_data)
            
            if not enrichment_result.is_reliable:
                self._logger.warning(
                    "LLM enrichment quality below threshold",
                    quality_score=enrichment_result.quality_metrics.overall_score,
                    confidence=enrichment_result.confidence_score
                )
                # Retourner données originales si qualité insuffisante
                return nutrition_data, {
                    "status": "quality_insufficient",
                    "quality_score": enrichment_result.quality_metrics.overall_score,
                    "confidence": enrichment_result.confidence_score
                }
            
            # Extraction des données enrichies
            enriched_analysis = enrichment_result.analysis
            
            # Mise à jour des données nutritionnelles avec enrichissement
            enriched_nutrition_data = NutritionData(
                # Données de base (conservées)
                product_name=nutrition_data.product_name,
                brand=nutrition_data.brand,
                barcode=nutrition_data.barcode,
                serving_size=nutrition_data.serving_size,
                calories=nutrition_data.calories,
                protein=nutrition_data.protein,
                carbohydrates=nutrition_data.carbohydrates,
                total_fat=nutrition_data.total_fat,
                saturated_fat=nutrition_data.saturated_fat,
                trans_fat=nutrition_data.trans_fat,
                fiber=nutrition_data.fiber,
                sugar=nutrition_data.sugar,
                sodium=nutrition_data.sodium,
                potassium=nutrition_data.potassium,
                cholesterol=nutrition_data.cholesterol,
                
                # Données enrichies par LLM
                ingredients=enriched_analysis.get("enhanced_ingredients", nutrition_data.ingredients or []),
                allergens=enriched_analysis.get("enhanced_allergens", nutrition_data.allergens or []),
                
                # Nouvelles informations LLM
                data_source=f"{nutrition_data.data_source}_llm_enriched",
                confidence_score=enrichment_result.confidence_score,
                
                # Ajout d'informations enrichies (si modèle le supporte)
                **{k: v for k, v in enriched_analysis.items() 
                   if k.startswith("enriched_") and hasattr(NutritionData, k)}
            )
            
            # Métadonnées d'enrichissement
            llm_metadata = {
                "status": "success",
                "provider_used": enrichment_result.provider_used.value,
                "quality_score": enrichment_result.quality_metrics.overall_score,
                "confidence_score": enrichment_result.confidence_score,
                "processing_time": enrichment_result.processing_time,
                "consensus_validated": enrichment_result.is_validated,
                "consensus_score": enrichment_result.consensus_score,
                "cache_hit": enrichment_result.cache_hit,
                "health_claims_validated": enrichment_result.health_claims_validated,
                "analysis_summary": enriched_analysis.get("analysis_summary", ""),
                "enhancement_details": {
                    "ingredients_enhanced": len(enriched_analysis.get("enhanced_ingredients", [])) > len(nutrition_data.ingredients or []),
                    "allergens_enhanced": len(enriched_analysis.get("enhanced_allergens", [])) > len(nutrition_data.allergens or []),
                    "nutrition_insights": enriched_analysis.get("nutrition_insights", {}),
                    "quality_warnings": enriched_analysis.get("quality_warnings", [])
                }
            }
            
            self._logger.info(
                "LLM enrichment completed successfully",
                quality_score=enrichment_result.quality_metrics.overall_score,
                confidence_score=enrichment_result.confidence_score,
                provider=enrichment_result.provider_used.value,
                processing_time=enrichment_result.processing_time,
                cache_hit=enrichment_result.cache_hit
            )
            
            return enriched_nutrition_data, llm_metadata
            
        except Exception as e:
            self._logger.error("LLM enrichment failed", error=str(e), product_name=nutrition_data.product_name)
            
            # En cas d'erreur, retourner données originales avec métadonnées d'erreur
            return nutrition_data, {
                "status": "error",
                "error": str(e),
                "fallback_used": True
            }


# ================================
# EXPERT ANALYSIS NODE
# ================================

class ExpertAnalysisNode(BaseWorkflowNode):
    """
    Node d'analyse experte - Strategy + Template Method Pattern.
    
    Effectue l'analyse de santé selon le niveau de qualité requis :
    - BASIC : Algorithmes cliniques uniquement
    - STANDARD : Léger enrichissement LLM
    - PREMIUM : Analyse LLM complète
    - EXPERT : Intelligence maximale avec contexte
    """
    
    def __init__(self, analysis_factory=None, llm_orchestrator=None):
        super().__init__()
        self._analysis_factory = analysis_factory  # Injected via DI
        self._llm_orchestrator = llm_orchestrator  # Injected via DI
    
    @property
    def node_name(self) -> str:
        return "expert_analysis"
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.EXPERT_ANALYSIS
    
    async def can_process(self, state: WorkflowState) -> bool:
        """Peut traiter si nous avons des données nutritionnelles."""
        return (state.current_stage == WorkflowStage.HEALTH_PROFILE_CONTEXT and
                state.nutrition_data is not None)
    
    async def validate_preconditions(self, state: WorkflowState) -> List[str]:
        """Valide les données nécessaires pour l'analyse."""
        errors = []
        
        if not state.nutrition_data:
            errors.append("Nutrition data required for analysis")
        
        if not state.user_context:
            errors.append("User health context required for analysis")
        
        return errors
    
    async def _process_internal(self, state: WorkflowState) -> WorkflowState:
        """Analyse experte selon le niveau de qualité."""
        quality_level = state.analysis_config.quality_level
        
        self._logger.info(
            "Starting expert analysis",
            workflow_id=state.workflow_id,
            quality_level=quality_level.value,
            product_name=state.nutrition_data.product_name
        )
        
        # Sélection de la stratégie d'analyse
        analysis_strategy = await self._select_analysis_strategy(state)
        
        # Exécution de l'analyse avec pipeline
        analysis_result = await self._execute_analysis_pipeline(
            analysis_strategy, 
            state.nutrition_data, 
            state.user_context
        )
        
        self._logger.info(
            "Expert analysis completed",
            workflow_id=state.workflow_id,
            overall_score=analysis_result.get("overall_score", 0),
            confidence=analysis_result.get("confidence", 0.0)
        )
        
        return state.with_expert_analysis(analysis_result)
    
    def _get_next_stage(self) -> Optional[WorkflowStage]:
        return WorkflowStage.SCORE_CALCULATION
    
    async def _select_analysis_strategy(self, state: WorkflowState):
        """Sélectionne la stratégie d'analyse selon la qualité."""
        quality_level = state.analysis_config.quality_level
        
        # Pour l'instant, retourne une stratégie basique
        # TODO: Implémenter de vraies stratégies d'analyse
        return {
            "name": f"{quality_level.value}_analysis",
            "requires_llm": quality_level in [QualityLevel.PREMIUM, QualityLevel.EXPERT],
            "quality_level": quality_level
        }
    
    async def _execute_analysis_pipeline(self, strategy, nutrition_data, user_context) -> Dict[str, Any]:
        """Exécute le pipeline d'analyse."""
        # Pipeline d'analyse en étapes
        pipeline_steps = [
            ("clinical_analysis", self._clinical_analysis),
            ("contextual_analysis", self._contextual_analysis),
            ("llm_enhancement", self._llm_enhancement),
            ("confidence_calculation", self._confidence_calculation)
        ]
        
        analysis_context = {
            "nutrition_data": nutrition_data,
            "user_context": user_context,
            "strategy": strategy
        }
        
        results = {}
        
        for step_name, step_func in pipeline_steps:
            try:
                step_result = await step_func(analysis_context)
                results[step_name] = step_result
                
                # Enrichissement du contexte pour les étapes suivantes
                analysis_context[f"{step_name}_result"] = step_result
                
            except Exception as e:
                self._logger.warning(
                    "Analysis pipeline step failed",
                    step=step_name,
                    error=str(e)
                )
                results[step_name] = {"error": str(e)}
        
        # Agrégation des résultats
        return self._aggregate_analysis_results(results)
    
    async def _clinical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse clinique algorithmique."""
        nutrition_data = context["nutrition_data"]
        user_context = context["user_context"]
        
        # Simulation d'analyse clinique
        scores = {
            "nutritional_quality": self._calculate_nutritional_quality(nutrition_data),
            "health_compatibility": self._calculate_health_compatibility(nutrition_data, user_context),
            "safety_assessment": self._calculate_safety_assessment(nutrition_data, user_context)
        }
        
        overall_score = sum(scores.values()) // len(scores)
        
        return {
            "overall_score": overall_score,
            "detailed_scores": scores,
            "recommendations": self._generate_clinical_recommendations(scores),
            "warnings": self._generate_clinical_warnings(scores)
        }
    
    async def _contextual_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse contextuelle basée sur le profil utilisateur."""
        user_context = context["user_context"]
        clinical_result = context.get("clinical_analysis_result", {})
        
        # Adaptation selon le contexte utilisateur
        context_factors = {
            "age_appropriateness": 0.8,  # Simulation
            "activity_level_match": 0.7,
            "dietary_preferences_match": 0.9,
            "health_goals_alignment": 0.6
        }
        
        context_score = sum(context_factors.values()) / len(context_factors) * 100
        
        return {
            "context_score": int(context_score),
            "context_factors": context_factors,
            "personalized_recommendations": [
                "Consider portion size based on your activity level",
                "This product aligns well with your dietary preferences",
                "Monitor sodium intake due to your health profile"
            ]
        }
    
    async def _llm_enhancement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichissement par LLM selon la stratégie."""
        strategy = context["strategy"]
        
        # Simulation d'enrichissement LLM
        if hasattr(strategy, "requires_llm") and strategy.requires_llm:
            return {
                "llm_insights": {
                    "enhanced_reasoning": "Advanced nutritional analysis reveals...",
                    "alternative_perspectives": ["Consider timing of consumption", "Pair with complementary foods"],
                    "scientific_context": "Recent studies suggest...",
                    "personalized_advice": "Based on your health profile..."
                },
                "model_used": "claude-3-sonnet",
                "confidence": 0.85
            }
        else:
            return {
                "llm_insights": None,
                "reason": "LLM enhancement not enabled for this quality level"
            }
    
    async def _confidence_calculation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule le score de confiance global."""
        clinical_result = context.get("clinical_analysis_result", {})
        contextual_result = context.get("contextual_analysis_result", {})
        llm_result = context.get("llm_enhancement_result", {})
        
        # Facteurs de confiance
        factors = {
            "data_completeness": 0.8,  # Basé sur la complétude des données nutritionnelles
            "clinical_confidence": 0.9,  # Confiance dans l'analyse clinique
            "context_relevance": 0.7,   # Pertinence du contexte utilisateur
            "llm_confidence": llm_result.get("confidence", 0.0)
        }
        
        overall_confidence = sum(factors.values()) / len(factors)
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_factors": factors,
            "reliability_score": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low"
        }
    
    def _aggregate_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Agrège tous les résultats d'analyse."""
        clinical = results.get("clinical_analysis", {})
        contextual = results.get("contextual_analysis", {})
        llm = results.get("llm_enhancement", {})
        confidence = results.get("confidence_calculation", {})
        
        return {
            "overall_score": clinical.get("overall_score", 0),
            "context_score": contextual.get("context_score", 0),
            "detailed_scores": clinical.get("detailed_scores", {}),
            "recommendations": clinical.get("recommendations", []) + contextual.get("personalized_recommendations", []),
            "warnings": clinical.get("warnings", []),
            "llm_insights": llm.get("llm_insights"),
            "confidence": confidence.get("overall_confidence", 0.0),
            "reliability": confidence.get("reliability_score", "medium"),
            "analysis_metadata": {
                "timestamp": time.time(),
                "model_used": llm.get("model_used"),
                "pipeline_steps": list(results.keys())
            }
        }
    
    def _calculate_nutritional_quality(self, nutrition_data: NutritionData) -> int:
        """Calcule la qualité nutritionnelle (simulation)."""
        score = 50  # Score de base
        
        # Facteurs positifs
        if nutrition_data.protein and nutrition_data.protein > 10:
            score += 10
        if nutrition_data.fiber and nutrition_data.fiber > 3:
            score += 10
        
        # Facteurs négatifs
        if nutrition_data.sugar and nutrition_data.sugar > 15:
            score -= 15
        if nutrition_data.sodium and nutrition_data.sodium > 400:
            score -= 10
        if nutrition_data.trans_fat and nutrition_data.trans_fat > 0:
            score -= 20
        
        return max(0, min(100, score))
    
    def _calculate_health_compatibility(self, nutrition_data: NutritionData, user_context: UserHealthContext) -> int:
        """Calcule la compatibilité avec le profil santé (simulation)."""
        # Simulation basée sur les conditions de santé
        base_score = 70
        
        # TODO: Intégrer la vraie logique des analyzers
        # Pour l'instant, simulation simple
        
        return base_score
    
    def _calculate_safety_assessment(self, nutrition_data: NutritionData, user_context: UserHealthContext) -> int:
        """Évalue la sécurité du produit (simulation)."""
        safety_score = 90  # Score de base élevé
        
        # Vérification des allergènes
        if nutrition_data.allergens:
            # TODO: Vérifier contre les allergies de l'utilisateur
            safety_score -= 10
        
        # Vérification des additifs
        if nutrition_data.additives:
            safety_score -= len(nutrition_data.additives) * 2
        
        return max(0, min(100, safety_score))
    
    def _generate_clinical_recommendations(self, scores: Dict[str, int]) -> List[str]:
        """Génère des recommandations cliniques."""
        recommendations = []
        
        if scores.get("nutritional_quality", 0) < 50:
            recommendations.append("Consider limiting consumption due to poor nutritional profile")
        
        if scores.get("health_compatibility", 0) < 60:
            recommendations.append("This product may not align well with your health profile")
        
        if scores.get("safety_assessment", 0) < 80:
            recommendations.append("Pay attention to allergens and additives in this product")
        
        return recommendations
    
    def _generate_clinical_warnings(self, scores: Dict[str, int]) -> List[str]:
        """Génère des avertissements cliniques."""
        warnings = []
        
        if scores.get("safety_assessment", 0) < 70:
            warnings.append("⚠️ Potential allergen risk detected")
        
        if all(score < 40 for score in scores.values()):
            warnings.append("⚠️ This product may not be suitable for your health profile")
        
        return warnings


# ================================
# MOCK STRATEGIES (pour les tests)
# ================================
