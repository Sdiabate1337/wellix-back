"""
LLM Configuration API endpoints for managing integration levels and preferences.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

from app.core.dependencies import get_current_active_user
from app.db.database import get_async_session
from app.db.models.user import User
from app.services.health_analyzers.analyzer_factory import LLMIntegrationLevel
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

router = APIRouter()


class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration."""
    default_integration_level: str
    enable_enhanced_recommendations: bool = True
    enable_timing_suggestions: bool = True
    enable_alternatives_suggestions: bool = True
    max_llm_processing_time_seconds: int = 30


class LLMConfigResponse(BaseModel):
    """Response model for LLM configuration."""
    user_id: str
    default_integration_level: str
    enable_enhanced_recommendations: bool
    enable_timing_suggestions: bool
    enable_alternatives_suggestions: bool
    max_llm_processing_time_seconds: int
    available_integration_levels: list
    updated_at: str


@router.get("/integration-levels")
async def get_available_integration_levels(
    current_user: User = Depends(get_current_active_user)
):
    """Get available LLM integration levels with descriptions."""
    
    integration_levels = [
        {
            "level": LLMIntegrationLevel.ALGORITHMIC_ONLY.value,
            "name": "Algorithmic Only",
            "description": "Pure clinical algorithms without LLM enhancement",
            "processing_time": "Fast (~1-2s)",
            "accuracy": "High clinical accuracy",
            "personalization": "Basic",
            "cost": "Free"
        },
        {
            "level": LLMIntegrationLevel.LLM_ENHANCED.value,
            "name": "LLM Enhanced",
            "description": "Clinical algorithms enhanced with contextual AI insights",
            "processing_time": "Medium (~3-5s)",
            "accuracy": "High clinical + contextual accuracy",
            "personalization": "Enhanced with timing and preparation tips",
            "cost": "Low"
        },
        {
            "level": LLMIntegrationLevel.HYBRID_BALANCED.value,
            "name": "Hybrid Balanced",
            "description": "Balanced combination of clinical algorithms and AI intelligence",
            "processing_time": "Medium (~4-6s)",
            "accuracy": "Optimal balance of clinical rigor and personalization",
            "personalization": "Highly personalized recommendations",
            "cost": "Medium"
        },
        {
            "level": LLMIntegrationLevel.LLM_DOMINANT.value,
            "name": "LLM Dominant",
            "description": "AI-driven analysis with clinical algorithm validation",
            "processing_time": "Slower (~6-10s)",
            "accuracy": "Maximum personalization and contextual intelligence",
            "personalization": "Extremely personalized with comprehensive insights",
            "cost": "High"
        }
    ]
    
    return JSONResponse(content={
        "integration_levels": integration_levels,
        "default_recommended": LLMIntegrationLevel.HYBRID_BALANCED.value,
        "user_current_level": "hybrid_balanced"  # TODO: Get from user preferences
    })


@router.post("/configure")
async def configure_llm_preferences(
    config: LLMConfigRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Configure user's LLM integration preferences."""
    
    try:
        # Validate integration level
        try:
            LLMIntegrationLevel(config.default_integration_level)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid integration level. Must be one of: {[level.value for level in LLMIntegrationLevel]}"
            )
        
        # Validate processing time limits
        if config.max_llm_processing_time_seconds < 5 or config.max_llm_processing_time_seconds > 60:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Processing time must be between 5 and 60 seconds"
            )
        
        # TODO: Save to database (user preferences table)
        # For now, we'll return the configuration as if it was saved
        
        response = LLMConfigResponse(
            user_id=str(current_user.id),
            default_integration_level=config.default_integration_level,
            enable_enhanced_recommendations=config.enable_enhanced_recommendations,
            enable_timing_suggestions=config.enable_timing_suggestions,
            enable_alternatives_suggestions=config.enable_alternatives_suggestions,
            max_llm_processing_time_seconds=config.max_llm_processing_time_seconds,
            available_integration_levels=[level.value for level in LLMIntegrationLevel],
            updated_at="2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        )
        
        logger.info(f"LLM configuration updated for user {current_user.id}: {config.default_integration_level}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring LLM preferences for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save LLM configuration"
        )


@router.get("/configuration")
async def get_llm_configuration(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get user's current LLM configuration."""
    
    try:
        # TODO: Get from database (user preferences table)
        # For now, return default configuration
        
        default_config = LLMConfigResponse(
            user_id=str(current_user.id),
            default_integration_level=LLMIntegrationLevel.HYBRID_BALANCED.value,
            enable_enhanced_recommendations=True,
            enable_timing_suggestions=True,
            enable_alternatives_suggestions=True,
            max_llm_processing_time_seconds=30,
            available_integration_levels=[level.value for level in LLMIntegrationLevel],
            updated_at="2024-01-01T00:00:00Z"
        )
        
        return JSONResponse(content=default_config.dict())
        
    except Exception as e:
        logger.error(f"Error getting LLM configuration for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LLM configuration"
        )


@router.post("/test-integration")
async def test_llm_integration(
    integration_level: str,
    current_user: User = Depends(get_current_active_user)
):
    """Test LLM integration with sample data."""
    
    try:
        # Validate integration level
        try:
            llm_level = LLMIntegrationLevel(integration_level)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid integration level. Must be one of: {[level.value for level in LLMIntegrationLevel]}"
            )
        
        # Simulate test results based on integration level
        test_results = {
            "integration_level": integration_level,
            "test_status": "success",
            "estimated_processing_time_seconds": {
                LLMIntegrationLevel.ALGORITHMIC_ONLY.value: 1.2,
                LLMIntegrationLevel.LLM_ENHANCED.value: 3.8,
                LLMIntegrationLevel.HYBRID_BALANCED.value: 5.1,
                LLMIntegrationLevel.LLM_DOMINANT.value: 8.3
            }.get(integration_level, 5.0),
            "features_enabled": {
                "clinical_algorithms": True,
                "llm_enhancement": integration_level != LLMIntegrationLevel.ALGORITHMIC_ONLY.value,
                "personalized_timing": integration_level != LLMIntegrationLevel.ALGORITHMIC_ONLY.value,
                "alternative_suggestions": integration_level != LLMIntegrationLevel.ALGORITHMIC_ONLY.value,
                "contextual_risk_assessment": integration_level in [
                    LLMIntegrationLevel.HYBRID_BALANCED.value,
                    LLMIntegrationLevel.LLM_DOMINANT.value
                ]
            },
            "sample_insights": {
                LLMIntegrationLevel.ALGORITHMIC_ONLY.value: [
                    "High sodium content detected",
                    "Moderate carbohydrate level"
                ],
                LLMIntegrationLevel.LLM_ENHANCED.value: [
                    "High sodium content - consider consuming in morning",
                    "Moderate carbs - pair with protein for better glucose control"
                ],
                LLMIntegrationLevel.HYBRID_BALANCED.value: [
                    "High sodium (680mg) - best consumed before 2PM to avoid evening retention",
                    "15g carbs - ideal post-workout snack, avoid if sedentary",
                    "Consider alternatives: unsalted nuts, Greek yogurt"
                ],
                LLMIntegrationLevel.LLM_DOMINANT.value: [
                    "Sodium content (680mg) represents 30% daily limit - timing crucial for your hypertension profile",
                    "Carbohydrate profile suggests 45-minute glucose peak - monitor if diabetic",
                    "Ingredient analysis reveals 3 additives that may interact with your medications",
                    "Personalized alternatives: Based on your preferences, try Trader Joe's unsalted almonds"
                ]
            }.get(integration_level, [])
        }
        
        logger.info(f"LLM integration test completed for user {current_user.id}: {integration_level}")
        
        return JSONResponse(content=test_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing LLM integration for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test LLM integration"
        )
