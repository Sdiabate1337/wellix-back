"""
Food analysis API endpoints for image processing and health analysis.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
import structlog
from datetime import datetime
import uuid

from app.core.dependencies import get_current_active_user, rate_limit_standard
from app.db.database import get_async_session
from app.db.models.user import User
from app.workflows.food_analysis_workflow import food_analysis_workflow
from app.models.health import UserHealthContext, NutritionData
from app.cache.cache_manager import cache_manager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.health_profile import UserHealthContext as DBUserHealthContext, HealthProfile
from app.db.models.food_analysis import FoodAnalysis

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/scan-food")
async def scan_food_product(
    image: UploadFile = File(...),
    barcode: Optional[str] = Form(None),
    current_user: User = Depends(rate_limit_standard),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Scan and analyze food product from image or barcode.
    
    This endpoint orchestrates the complete food analysis workflow:
    1. OCR extraction from uploaded image
    2. Barcode lookup in OpenFoodFacts
    3. Multi-profile health analysis
    4. Personalized recommendations generation
    """
    try:
        # Validate image file
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format. Please upload JPEG, PNG, or WebP images."
            )
        
        # Read image data
        image_data = await image.read()
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image file too large. Maximum size is 10MB."
            )
        
        # Get user health context
        user_context = await _get_user_health_context(current_user.id, db)
        
        # Execute workflow
        workflow_result = await food_analysis_workflow.process_food_analysis(
            image_data=image_data,
            barcode=barcode,
            user_context=user_context
        )
        
        # Check for workflow errors
        if workflow_result.get("errors"):
            logger.error(f"Workflow errors for user {current_user.id}: {workflow_result['errors']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Analysis failed. Please try again with a clearer image."
            )
        
        # Save analysis to database
        analysis_id = await _save_analysis_result(workflow_result, current_user.id, db)
        
        # Prepare response
        response_data = {
            "analysis_id": str(analysis_id),
            "product_name": workflow_result.get("nutrition_data", {}).get("product_name", "Unknown Product"),
            "overall_score": workflow_result.get("overall_score", 0),
            "safety_level": workflow_result.get("safety_level", "unknown"),
            "recommendations": workflow_result.get("recommendations", []),
            "health_analysis": workflow_result.get("health_analysis", {}),
            "confidence_score": workflow_result.get("confidence_score", 0.5),
            "processing_time_ms": workflow_result.get("processing_time_ms", 0),
            "chat_context": workflow_result.get("chat_context", {}),
            "analysis_summary": workflow_result.get("analysis_summary", "")
        }
        
        logger.info(f"Food analysis completed for user {current_user.id}: {analysis_id}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Food analysis error for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed due to an internal error. Please try again."
        )


@router.get("/analysis/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get detailed analysis result by ID."""
    try:
        # Check cache first
        cached_result = await cache_manager.get_food_analysis(analysis_id)
        if cached_result:
            return JSONResponse(content=cached_result)
        
        # Get from database
        result = await db.execute(
            select(FoodAnalysis).where(
                FoodAnalysis.id == analysis_id,
                FoodAnalysis.user_id == current_user.id
            )
        )
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Convert to response format
        response_data = {
            "analysis_id": str(analysis.id),
            "product_name": analysis.product_name,
            "brand": analysis.brand,
            "barcode": analysis.barcode,
            "overall_score": analysis.overall_score,
            "safety_level": "safe" if analysis.overall_score >= 70 else "caution",
            "nutrition_data": {
                "serving_size": analysis.serving_size,
                "calories": analysis.calories,
                "protein": analysis.protein,
                "carbohydrates": analysis.carbohydrates,
                "total_fat": analysis.total_fat,
                "saturated_fat": analysis.saturated_fat,
                "fiber": analysis.fiber,
                "sugar": analysis.sugar,
                "sodium": analysis.sodium,
                "ingredients": analysis.ingredients,
                "allergens": analysis.allergens
            },
            "profile_scores": analysis.profile_scores,
            "recommendations": analysis.recommendations,
            "warnings": analysis.warnings,
            "created_at": analysis.created_at.isoformat()
        }
        
        # Cache the result
        await cache_manager.set_food_analysis(analysis_id, response_data)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )


@router.get("/history")
async def get_analysis_history(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get user's analysis history."""
    try:
        result = await db.execute(
            select(FoodAnalysis)
            .where(FoodAnalysis.user_id == current_user.id)
            .order_by(FoodAnalysis.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        analyses = result.scalars().all()
        
        history_items = []
        for analysis in analyses:
            history_items.append({
                "analysis_id": str(analysis.id),
                "product_name": analysis.product_name,
                "brand": analysis.brand,
                "overall_score": analysis.overall_score,
                "created_at": analysis.created_at.isoformat(),
                "safety_level": "safe" if analysis.overall_score >= 70 else "caution"
            })
        
        return JSONResponse(content={
            "history": history_items,
            "total": len(history_items),
            "limit": limit,
            "offset": offset
        })
        
    except Exception as e:
        logger.error(f"Error retrieving history for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis history"
        )


@router.post("/validate-image")
async def validate_nutrition_image(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Validate if uploaded image contains a nutrition label."""
    try:
        # Validate image file
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        image_data = await image.read()
        
        # Use OCR service to validate
        from app.services.ocr_service import ocr_service
        is_valid, reason = await ocr_service.validate_nutrition_label(image_data)
        
        return JSONResponse(content={
            "is_valid": is_valid,
            "reason": reason,
            "suggestions": [
                "Ensure the nutrition label is clearly visible",
                "Make sure the image is well-lit and in focus",
                "Try to capture the entire nutrition facts panel"
            ] if not is_valid else []
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate image"
        )


async def _get_user_health_context(user_id: str, db: AsyncSession) -> UserHealthContext:
    """Get user's health context from database."""
    try:
        # Check cache first
        cached_context = await cache_manager.get_user_health_context(str(user_id))
        if cached_context:
            return UserHealthContext(**cached_context)
        
        # Get from database
        context_result = await db.execute(
            select(DBUserHealthContext).where(DBUserHealthContext.user_id == user_id)
        )
        db_context = context_result.scalar_one_or_none()
        
        profiles_result = await db.execute(
            select(HealthProfile).where(HealthProfile.user_id == user_id)
        )
        db_profiles = profiles_result.scalars().all()
        
        # Convert to Pydantic model
        from app.models.health import HealthProfile as PydanticHealthProfile, ProfileType, Severity, AgeGroup, ActivityLevel
        
        profiles = []
        for db_profile in db_profiles:
            profiles.append(PydanticHealthProfile(
                profile_type=ProfileType(db_profile.profile_type),
                severity=Severity(db_profile.severity),
                restrictions=db_profile.restrictions or [],
                goals=db_profile.goals or [],
                medications=db_profile.medications or [],
                is_primary=db_profile.is_primary,
                notes=db_profile.notes,
                target_values=db_profile.target_values or {}
            ))
        
        if db_context:
            user_context = UserHealthContext(
                user_id=str(user_id),
                primary_profiles=profiles,
                allergies=db_context.allergies or [],
                dietary_preferences=db_context.dietary_preferences or [],
                age_group=AgeGroup(db_context.age_group),
                activity_level=ActivityLevel(db_context.activity_level),
                weight_goals=db_context.weight_goals,
                height_cm=db_context.height_cm,
                weight_kg=db_context.weight_kg,
                preferred_language=db_context.preferred_language,
                analysis_depth=db_context.analysis_depth
            )
        else:
            # Default context for new users
            user_context = UserHealthContext(
                user_id=str(user_id),
                primary_profiles=profiles,
                age_group=AgeGroup.ADULT,
                activity_level=ActivityLevel.MODERATELY_ACTIVE
            )
        
        # Cache the context
        await cache_manager.set_user_health_context(str(user_id), user_context.dict())
        
        return user_context
        
    except Exception as e:
        logger.error(f"Error getting user health context: {e}")
        # Return minimal default context
        return UserHealthContext(
            user_id=str(user_id),
            age_group=AgeGroup.ADULT,
            activity_level=ActivityLevel.MODERATELY_ACTIVE
        )


async def _save_analysis_result(workflow_result: Dict[str, Any], user_id: str, db: AsyncSession) -> uuid.UUID:
    """Save analysis result to database."""
    try:
        nutrition_data = workflow_result.get("nutrition_data", {})
        health_analysis = workflow_result.get("health_analysis", {})
        
        analysis = FoodAnalysis(
            user_id=user_id,
            product_name=nutrition_data.get("product_name", "Unknown Product"),
            brand=nutrition_data.get("brand"),
            barcode=nutrition_data.get("barcode"),
            serving_size=nutrition_data.get("serving_size", "1 serving"),
            calories=nutrition_data.get("calories", 0),
            protein=nutrition_data.get("protein", 0),
            carbohydrates=nutrition_data.get("carbohydrates", 0),
            total_fat=nutrition_data.get("total_fat", 0),
            saturated_fat=nutrition_data.get("saturated_fat"),
            fiber=nutrition_data.get("fiber"),
            sugar=nutrition_data.get("sugar"),
            sodium=nutrition_data.get("sodium"),
            potassium=nutrition_data.get("potassium"),
            cholesterol=nutrition_data.get("cholesterol"),
            ingredients=nutrition_data.get("ingredients", []),
            allergens=nutrition_data.get("allergens", []),
            additives=nutrition_data.get("additives", []),
            overall_score=workflow_result.get("overall_score", 0),
            safety_score=min(workflow_result.get("overall_score", 0) + 10, 100),
            profile_scores=health_analysis.get("profile_results", {}),
            recommendations=workflow_result.get("recommendations", []),
            warnings=[],  # Warnings are embedded in profile_scores
            confidence_score=workflow_result.get("confidence_score", 0.5),
            processing_time_ms=workflow_result.get("processing_time_ms"),
            data_source="workflow"
        )
        
        db.add(analysis)
        await db.commit()
        await db.refresh(analysis)
        
        return analysis.id
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error saving analysis result: {e}")
        raise
