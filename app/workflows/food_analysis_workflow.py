"""
LangGraph workflow for orchestrating the complete food analysis pipeline.
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph import StateGraph, END
from langgraph.graph import Graph
import structlog
import asyncio
from datetime import datetime

from app.models.health import NutritionData, UserHealthContext, AnalysisResult
from app.services.ocr_service import ocr_service
from app.services.openfoodfacts_service import openfoodfacts_service
from app.services.health_analyzers.analyzer_factory import HealthAnalyzerFactory
from app.cache.cache_manager import cache_manager

logger = structlog.get_logger(__name__)


class WellixState(TypedDict):
    """State object for the Wellix food analysis workflow."""
    
    # Input data
    image_data: Optional[bytes]
    barcode: Optional[str]
    user_context: UserHealthContext
    
    # Processing results
    ocr_result: Optional[Dict[str, Any]]
    openfoodfacts_data: Optional[Dict[str, Any]]
    nutrition_data: Optional[NutritionData]
    
    # Analysis results
    health_analysis: Optional[Dict[str, Any]]
    overall_score: Optional[int]
    safety_level: Optional[str]
    recommendations: Optional[List[str]]
    
    # Workflow metadata
    processing_steps: List[str]
    errors: List[str]
    processing_time_ms: Optional[int]
    confidence_score: Optional[float]
    
    # Chat context
    analysis_summary: Optional[str]
    chat_context: Optional[Dict[str, Any]]


class FoodAnalysisWorkflow:
    """LangGraph workflow for complete food analysis pipeline."""
    
    def __init__(self):
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> Graph:
        """Build the LangGraph workflow graph."""
        
        # Create workflow graph
        workflow = StateGraph(WellixState)
        
        # Add nodes
        workflow.add_node("ocr_extraction", self._ocr_extraction_node)
        workflow.add_node("barcode_lookup", self._barcode_lookup_node)
        workflow.add_node("data_enrichment", self._data_enrichment_node)
        workflow.add_node("nutrition_parsing", self._nutrition_parsing_node)
        workflow.add_node("health_analysis", self._health_analysis_node)
        workflow.add_node("recommendation_generation", self._recommendation_generation_node)
        workflow.add_node("chat_context_preparation", self._chat_context_preparation_node)
        
        # Define workflow edges
        workflow.set_entry_point("ocr_extraction")
        
        # OCR -> Barcode lookup (if barcode found) or Data enrichment
        workflow.add_conditional_edges(
            "ocr_extraction",
            self._route_after_ocr,
            {
                "barcode_lookup": "barcode_lookup",
                "data_enrichment": "data_enrichment",
                "error": END
            }
        )
        
        # Barcode lookup -> Data enrichment
        workflow.add_edge("barcode_lookup", "data_enrichment")
        
        # Data enrichment -> Nutrition parsing
        workflow.add_edge("data_enrichment", "nutrition_parsing")
        
        # Nutrition parsing -> Health analysis
        workflow.add_edge("nutrition_parsing", "health_analysis")
        
        # Health analysis -> Recommendation generation
        workflow.add_edge("health_analysis", "recommendation_generation")
        
        # Recommendation generation -> Chat context preparation
        workflow.add_edge("recommendation_generation", "chat_context_preparation")
        
        # Chat context preparation -> END
        workflow.add_edge("chat_context_preparation", END)
        
        return workflow.compile()
    
    async def process_food_analysis(
        self,
        image_data: Optional[bytes] = None,
        barcode: Optional[str] = None,
        user_context: UserHealthContext = None
    ) -> WellixState:
        """
        Process complete food analysis workflow.
        
        Args:
            image_data: Raw image bytes for OCR processing
            barcode: Product barcode if available
            user_context: User's health context and preferences
            
        Returns:
            Complete workflow state with analysis results
        """
        start_time = datetime.utcnow()
        
        # Initialize state
        initial_state: WellixState = {
            "image_data": image_data,
            "barcode": barcode,
            "user_context": user_context,
            "ocr_result": None,
            "openfoodfacts_data": None,
            "nutrition_data": None,
            "health_analysis": None,
            "overall_score": None,
            "safety_level": None,
            "recommendations": None,
            "processing_steps": [],
            "errors": [],
            "processing_time_ms": None,
            "confidence_score": None,
            "analysis_summary": None,
            "chat_context": None
        }
        
        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate total processing time
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            final_state["processing_time_ms"] = processing_time
            
            logger.info(f"Food analysis workflow completed in {processing_time}ms")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state["errors"].append(f"Workflow execution failed: {str(e)}")
            return initial_state
    
    async def _ocr_extraction_node(self, state: WellixState) -> WellixState:
        """Extract text and nutrition data from image using OCR."""
        state["processing_steps"].append("ocr_extraction")
        
        if not state["image_data"]:
            if not state["barcode"]:
                state["errors"].append("No image data or barcode provided")
                return state
            # Skip OCR if we have barcode
            return state
        
        try:
            # Perform OCR extraction
            ocr_result = await ocr_service.extract_text_from_image(state["image_data"])
            state["ocr_result"] = ocr_result
            
            # Extract barcode from OCR if not provided
            if not state["barcode"] and ocr_result.get("structured_data", {}).get("barcode"):
                state["barcode"] = ocr_result["structured_data"]["barcode"]
            
            logger.info("OCR extraction completed successfully")
            
        except Exception as e:
            error_msg = f"OCR extraction failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _barcode_lookup_node(self, state: WellixState) -> WellixState:
        """Look up product data using barcode."""
        state["processing_steps"].append("barcode_lookup")
        
        if not state["barcode"]:
            return state
        
        try:
            # Look up product in OpenFoodFacts
            product_data = await openfoodfacts_service.get_product_by_barcode(state["barcode"])
            
            if product_data:
                state["openfoodfacts_data"] = product_data
                logger.info(f"Product data found for barcode {state['barcode']}")
            else:
                logger.info(f"No product data found for barcode {state['barcode']}")
            
        except Exception as e:
            error_msg = f"Barcode lookup failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _data_enrichment_node(self, state: WellixState) -> WellixState:
        """Enrich and merge data from multiple sources."""
        state["processing_steps"].append("data_enrichment")
        
        try:
            # Merge OCR and OpenFoodFacts data
            enriched_data = self._merge_nutrition_data(
                state.get("ocr_result"),
                state.get("openfoodfacts_data")
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_data_confidence(
                state.get("ocr_result"),
                state.get("openfoodfacts_data")
            )
            state["confidence_score"] = confidence_score
            
            # Store enriched data for next step
            state["enriched_nutrition_data"] = enriched_data
            
            logger.info("Data enrichment completed")
            
        except Exception as e:
            error_msg = f"Data enrichment failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _nutrition_parsing_node(self, state: WellixState) -> WellixState:
        """Parse and validate nutrition data."""
        state["processing_steps"].append("nutrition_parsing")
        
        try:
            enriched_data = state.get("enriched_nutrition_data", {})
            
            # Create NutritionData object
            nutrition_data = self._create_nutrition_data_object(enriched_data)
            state["nutrition_data"] = nutrition_data
            
            logger.info("Nutrition data parsing completed")
            
        except Exception as e:
            error_msg = f"Nutrition parsing failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _health_analysis_node(self, state: WellixState) -> WellixState:
        """Perform health analysis based on user profiles."""
        state["processing_steps"].append("health_analysis")
        
        if not state["nutrition_data"] or not state["user_context"]:
            state["errors"].append("Missing nutrition data or user context for health analysis")
            return state
        
        try:
            # Perform multi-profile health analysis
            analysis_results = await HealthAnalyzerFactory.analyze_for_user(
                state["nutrition_data"],
                state["user_context"]
            )
            
            # Calculate overall score
            overall_score = HealthAnalyzerFactory.get_overall_score(analysis_results)
            
            # Determine overall safety level
            safety_levels = [result.safety_level for result in analysis_results.values()]
            overall_safety = self._determine_overall_safety(safety_levels)
            
            state["health_analysis"] = {
                "profile_results": {k: self._serialize_analysis_result(v) for k, v in analysis_results.items()},
                "overall_score": overall_score,
                "safety_level": overall_safety
            }
            state["overall_score"] = overall_score
            state["safety_level"] = overall_safety
            
            logger.info(f"Health analysis completed with overall score: {overall_score}")
            
        except Exception as e:
            error_msg = f"Health analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _recommendation_generation_node(self, state: WellixState) -> WellixState:
        """Generate personalized recommendations."""
        state["processing_steps"].append("recommendation_generation")
        
        if not state["health_analysis"]:
            state["errors"].append("No health analysis available for recommendation generation")
            return state
        
        try:
            # Collect recommendations from all profile analyses
            all_recommendations = []
            profile_results = state["health_analysis"]["profile_results"]
            
            for profile_type, result in profile_results.items():
                all_recommendations.extend(result.get("recommendations", []))
            
            # Remove duplicates while preserving order
            unique_recommendations = []
            seen = set()
            for rec in all_recommendations:
                if rec not in seen:
                    unique_recommendations.append(rec)
                    seen.add(rec)
            
            state["recommendations"] = unique_recommendations[:10]  # Limit to top 10
            
            logger.info(f"Generated {len(unique_recommendations)} recommendations")
            
        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _chat_context_preparation_node(self, state: WellixState) -> WellixState:
        """Prepare context for AI chat integration."""
        state["processing_steps"].append("chat_context_preparation")
        
        try:
            # Create analysis summary
            analysis_summary = self._create_analysis_summary(state)
            state["analysis_summary"] = analysis_summary
            
            # Prepare chat context
            chat_context = {
                "product_name": state.get("nutrition_data", {}).get("product_name", "Unknown Product"),
                "overall_score": state.get("overall_score", 0),
                "safety_level": state.get("safety_level", "unknown"),
                "key_recommendations": state.get("recommendations", [])[:5],
                "health_profiles": list(state.get("health_analysis", {}).get("profile_results", {}).keys()),
                "analysis_summary": analysis_summary,
                "confidence_score": state.get("confidence_score", 0.5)
            }
            
            state["chat_context"] = chat_context
            
            logger.info("Chat context preparation completed")
            
        except Exception as e:
            error_msg = f"Chat context preparation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def _route_after_ocr(self, state: WellixState) -> str:
        """Route workflow after OCR based on available data."""
        if state.get("errors"):
            return "error"
        
        if state.get("barcode"):
            return "barcode_lookup"
        else:
            return "data_enrichment"
    
    def _merge_nutrition_data(
        self,
        ocr_result: Optional[Dict[str, Any]],
        openfoodfacts_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge nutrition data from OCR and OpenFoodFacts."""
        merged_data = {}
        
        # Start with OpenFoodFacts data (more reliable)
        if openfoodfacts_data:
            merged_data.update(openfoodfacts_data)
        
        # Supplement with OCR data where missing
        if ocr_result and ocr_result.get("structured_data"):
            ocr_nutrition = ocr_result["structured_data"]
            
            for key, value in ocr_nutrition.items():
                if key not in merged_data or merged_data[key] is None:
                    merged_data[key] = value
        
        return merged_data
    
    def _calculate_data_confidence(
        self,
        ocr_result: Optional[Dict[str, Any]],
        openfoodfacts_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for merged data."""
        confidence = 0.0
        
        # OpenFoodFacts data adds high confidence
        if openfoodfacts_data:
            data_quality = openfoodfacts_data.get("data_quality", "low")
            if data_quality == "high":
                confidence += 0.7
            elif data_quality == "medium":
                confidence += 0.5
            else:
                confidence += 0.3
        
        # OCR data adds moderate confidence
        if ocr_result:
            ocr_confidence = ocr_result.get("confidence_scores", {}).get("overall", 0.0)
            confidence += ocr_confidence * 0.3
        
        return min(confidence, 1.0)
    
    def _create_nutrition_data_object(self, enriched_data: Dict[str, Any]) -> NutritionData:
        """Create NutritionData object from enriched data."""
        # Extract nutrition values with defaults
        nutrition_data = enriched_data.get("nutrition_data", {})
        
        return NutritionData(
            product_name=enriched_data.get("product_name", "Unknown Product"),
            brand=enriched_data.get("brand", ""),
            barcode=enriched_data.get("barcode"),
            serving_size=enriched_data.get("serving_size", "1 serving"),
            calories=nutrition_data.get("calories_per_serving", nutrition_data.get("calories_per_100g", 0)),
            protein=nutrition_data.get("protein_per_serving", nutrition_data.get("protein_per_100g", 0)),
            carbohydrates=nutrition_data.get("carbohydrates_per_serving", nutrition_data.get("carbohydrates_per_100g", 0)),
            total_fat=nutrition_data.get("total_fat_per_serving", nutrition_data.get("total_fat_per_100g", 0)),
            saturated_fat=nutrition_data.get("saturated_fat_per_serving", nutrition_data.get("saturated_fat_per_100g")),
            trans_fat=nutrition_data.get("trans_fat_per_serving", nutrition_data.get("trans_fat_per_100g")),
            fiber=nutrition_data.get("fiber_per_serving", nutrition_data.get("fiber_per_100g")),
            sugar=nutrition_data.get("sugar_per_serving", nutrition_data.get("sugar_per_100g")),
            sodium=nutrition_data.get("sodium_per_serving", nutrition_data.get("sodium_per_100g")),
            potassium=nutrition_data.get("potassium_per_serving", nutrition_data.get("potassium_per_100g")),
            cholesterol=nutrition_data.get("cholesterol_per_serving", nutrition_data.get("cholesterol_per_100g")),
            ingredients=enriched_data.get("ingredients", []),
            allergens=enriched_data.get("allergens", []),
            additives=enriched_data.get("additives", []),
            data_source="workflow_merged"
        )
    
    def _serialize_analysis_result(self, result: AnalysisResult) -> Dict[str, Any]:
        """Serialize AnalysisResult for JSON storage."""
        return {
            "score": result.score,
            "safety_level": result.safety_level,
            "recommendations": result.recommendations,
            "warnings": result.warnings,
            "detailed_scores": result.detailed_scores,
            "reasoning": result.reasoning
        }
    
    def _determine_overall_safety(self, safety_levels: List[str]) -> str:
        """Determine overall safety level from multiple profile results."""
        if "danger" in safety_levels:
            return "danger"
        elif "warning" in safety_levels:
            return "warning"
        elif "caution" in safety_levels:
            return "caution"
        else:
            return "safe"
    
    def _create_analysis_summary(self, state: WellixState) -> str:
        """Create human-readable analysis summary."""
        product_name = state.get("nutrition_data", {}).get("product_name", "this product")
        overall_score = state.get("overall_score", 0)
        safety_level = state.get("safety_level", "unknown")
        
        summary_parts = [
            f"Analysis of {product_name}:",
            f"Overall health score: {overall_score}/100",
            f"Safety level: {safety_level.title()}"
        ]
        
        # Add key concerns
        health_analysis = state.get("health_analysis", {})
        profile_results = health_analysis.get("profile_results", {})
        
        key_concerns = []
        for profile_type, result in profile_results.items():
            warnings = result.get("warnings", [])
            critical_warnings = [w for w in warnings if w.get("severity") == "critical"]
            if critical_warnings:
                key_concerns.extend([w.get("message", "") for w in critical_warnings])
        
        if key_concerns:
            summary_parts.append("Key concerns: " + "; ".join(key_concerns[:3]))
        
        return "\n".join(summary_parts)


# Global workflow instance
food_analysis_workflow = FoodAnalysisWorkflow()
