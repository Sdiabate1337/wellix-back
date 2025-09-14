"""
Nutrition Analysis AI Service
Provides AI-enhanced nutrition analysis using LLM integration.
"""

from typing import Dict, Any, Optional
import structlog
from app.services.ai.ai_service import AIServiceManager, NutritionAnalysisAI

logger = structlog.get_logger(__name__)

# Global AI service instance
_ai_manager = None
_nutrition_ai = None

def _get_ai_service():
    """Get or create AI service instance."""
    global _ai_manager, _nutrition_ai
    
    if _ai_manager is None:
        _ai_manager = AIServiceManager()
        _nutrition_ai = NutritionAnalysisAI(_ai_manager)
    
    return _nutrition_ai

async def enhance_nutrition_analysis(
    nutrition_data: Dict[str, Any],
    health_context: Dict[str, Any],
    analysis_prompt: str,
    insight_level: str = "balanced"
) -> Dict[str, Any]:
    """
    Enhance nutrition analysis with AI insights.
    
    Args:
        nutrition_data: Nutrition information from the product
        health_context: User's health context and conditions
        analysis_prompt: Specific prompt for the analysis
        insight_level: Level of AI insight needed
        
    Returns:
        Enhanced analysis with AI insights
    """
    try:
        ai_service = _get_ai_service()
        
        # Create enhanced context for AI analysis
        enhanced_context = {
            "nutrition": nutrition_data,
            "health": health_context,
            "prompt": analysis_prompt,
            "insight_level": insight_level
        }
        
        # Get AI enhancement
        result = await ai_service.enhance_nutrition_analysis(
            nutrition_data=nutrition_data,
            health_context=health_context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI nutrition analysis failed: {e}")
        # Return mock response for testing
        return {
            "ai_insights": {
                "enhanced_score": 0,
                "reasoning": "AI service unavailable - using clinical analysis only",
                "recommendations": [],
                "alternatives": [],
                "timing_advice": "",
                "risk_factors": []
            },
            "model_used": "mock",
            "usage": {"tokens": 0},
            "error": str(e)
        }

# Mock function for testing without AI services
async def mock_enhance_nutrition_analysis(
    nutrition_data: Dict[str, Any],
    health_context: Dict[str, Any],
    analysis_prompt: str,
    insight_level: str = "balanced"
) -> Dict[str, Any]:
    """
    Mock AI enhancement for testing purposes.
    """
    # Simulate AI insights based on the data
    product_name = nutrition_data.get("product_name", "Unknown Product")
    calories = nutrition_data.get("calories", 0)
    sugar = nutrition_data.get("sugar", 0)
    
    # Mock reasoning based on product characteristics
    if sugar > 15:
        risk_level = "high"
        score_adjustment = -20
        reasoning = f"{product_name} has high sugar content ({sugar}g), which may cause blood sugar spikes."
    elif calories > 300:
        risk_level = "moderate"
        score_adjustment = -10
        reasoning = f"{product_name} is calorie-dense ({calories} cal), consider portion control."
    else:
        risk_level = "low"
        score_adjustment = 5
        reasoning = f"{product_name} appears to be a reasonable nutritional choice."
    
    return {
        "ai_insights": {
            "enhanced_score": score_adjustment,
            "reasoning": reasoning,
            "recommendations": [
                f"Consider consuming {product_name} in moderation",
                "Pair with protein or fiber to slow absorption"
            ],
            "alternatives": [
                "Fresh fruit for natural sweetness",
                "Nuts for healthy fats and protein"
            ],
            "timing_advice": "Best consumed post-workout or as part of balanced meal",
            "risk_factors": [risk_level],
            "confidence": 0.75
        },
        "model_used": "mock-ai",
        "usage": {"tokens": 150}
    }
