"""
Health profile management API endpoints for user health conditions and preferences.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import structlog
from datetime import datetime

from app.core.dependencies import get_current_active_user
from app.db.database import get_async_session
from app.db.models.user import User
from app.db.models.health_profile import HealthProfile, UserHealthContext as DBUserHealthContext
from app.models.health import ProfileType, Severity, AgeGroup, ActivityLevel
from app.cache.cache_manager import cache_manager
from pydantic import BaseModel, validator

logger = structlog.get_logger(__name__)

router = APIRouter()


class HealthProfileCreate(BaseModel):
    profile_type: ProfileType
    severity: Severity
    restrictions: List[str] = []
    goals: List[str] = []
    medications: List[str] = []
    is_primary: bool = False
    notes: Optional[str] = None
    target_values: Dict[str, float] = {}


class HealthProfileUpdate(BaseModel):
    severity: Optional[Severity] = None
    restrictions: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    is_primary: Optional[bool] = None
    notes: Optional[str] = None
    target_values: Optional[Dict[str, float]] = None


class UserHealthContextUpdate(BaseModel):
    age_group: Optional[AgeGroup] = None
    activity_level: Optional[ActivityLevel] = None
    weight_goals: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    allergies: Optional[List[str]] = None
    dietary_preferences: Optional[List[str]] = None
    analysis_depth: Optional[str] = None
    preferred_language: Optional[str] = None


@router.post("/profiles")
async def create_health_profile(
    profile_data: HealthProfileCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Create a new health profile for the user."""
    try:
        # Check if profile type already exists
        result = await db.execute(
            select(HealthProfile).where(
                HealthProfile.user_id == current_user.id,
                HealthProfile.profile_type == profile_data.profile_type.value
            )
        )
        existing_profile = result.scalar_one_or_none()
        
        if existing_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Health profile for {profile_data.profile_type.value} already exists"
            )
        
        # Create new profile
        new_profile = HealthProfile(
            user_id=current_user.id,
            profile_type=profile_data.profile_type.value,
            severity=profile_data.severity.value,
            restrictions=profile_data.restrictions,
            goals=profile_data.goals,
            medications=profile_data.medications,
            is_primary=profile_data.is_primary,
            notes=profile_data.notes,
            target_values=profile_data.target_values
        )
        
        db.add(new_profile)
        await db.commit()
        await db.refresh(new_profile)
        
        # Invalidate user cache
        await cache_manager.invalidate_user_cache(str(current_user.id))
        
        logger.info(f"Health profile created for user {current_user.id}: {profile_data.profile_type.value}")
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "profile_id": str(new_profile.id),
                "profile_type": new_profile.profile_type,
                "severity": new_profile.severity,
                "is_primary": new_profile.is_primary,
                "created_at": new_profile.created_at.isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating health profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create health profile"
        )


@router.get("/profiles")
async def get_health_profiles(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get all health profiles for the current user."""
    try:
        result = await db.execute(
            select(HealthProfile).where(HealthProfile.user_id == current_user.id)
        )
        profiles = result.scalars().all()
        
        profile_list = []
        for profile in profiles:
            profile_list.append({
                "profile_id": str(profile.id),
                "profile_type": profile.profile_type,
                "severity": profile.severity,
                "restrictions": profile.restrictions or [],
                "goals": profile.goals or [],
                "medications": profile.medications or [],
                "is_primary": profile.is_primary,
                "notes": profile.notes,
                "target_values": profile.target_values or {},
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            })
        
        return JSONResponse(content={"profiles": profile_list})
        
    except Exception as e:
        logger.error(f"Error retrieving health profiles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health profiles"
        )


@router.put("/profiles/{profile_id}")
async def update_health_profile(
    profile_id: str,
    profile_data: HealthProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Update an existing health profile."""
    try:
        # Get profile
        result = await db.execute(
            select(HealthProfile).where(
                HealthProfile.id == profile_id,
                HealthProfile.user_id == current_user.id
            )
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health profile not found"
            )
        
        # Update fields
        update_data = profile_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field in ["severity"] and value:
                setattr(profile, field, value.value)
            else:
                setattr(profile, field, value)
        
        profile.updated_at = datetime.utcnow()
        await db.commit()
        
        # Invalidate user cache
        await cache_manager.invalidate_user_cache(str(current_user.id))
        
        logger.info(f"Health profile updated: {profile_id}")
        
        return JSONResponse(content={"message": "Health profile updated successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating health profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update health profile"
        )


@router.delete("/profiles/{profile_id}")
async def delete_health_profile(
    profile_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Delete a health profile."""
    try:
        # Verify ownership and delete
        result = await db.execute(
            delete(HealthProfile).where(
                HealthProfile.id == profile_id,
                HealthProfile.user_id == current_user.id
            )
        )
        
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health profile not found"
            )
        
        await db.commit()
        
        # Invalidate user cache
        await cache_manager.invalidate_user_cache(str(current_user.id))
        
        logger.info(f"Health profile deleted: {profile_id}")
        
        return JSONResponse(content={"message": "Health profile deleted successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting health profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete health profile"
        )


@router.get("/context")
async def get_health_context(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get user's complete health context."""
    try:
        # Get health context
        result = await db.execute(
            select(DBUserHealthContext).where(DBUserHealthContext.user_id == current_user.id)
        )
        context = result.scalar_one_or_none()
        
        if not context:
            # Return default context
            return JSONResponse(content={
                "age_group": "adult",
                "activity_level": "moderately_active",
                "allergies": [],
                "dietary_preferences": [],
                "analysis_depth": "standard",
                "preferred_language": "en"
            })
        
        return JSONResponse(content={
            "age_group": context.age_group,
            "activity_level": context.activity_level,
            "weight_goals": context.weight_goals,
            "height_cm": context.height_cm,
            "weight_kg": context.weight_kg,
            "bmi": context.bmi,
            "allergies": context.allergies or [],
            "dietary_preferences": context.dietary_preferences or [],
            "analysis_depth": context.analysis_depth,
            "preferred_language": context.preferred_language,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error retrieving health context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health context"
        )


@router.put("/context")
async def update_health_context(
    context_data: UserHealthContextUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Update user's health context."""
    try:
        # Get or create health context
        result = await db.execute(
            select(DBUserHealthContext).where(DBUserHealthContext.user_id == current_user.id)
        )
        context = result.scalar_one_or_none()
        
        if not context:
            # Create new context
            context = DBUserHealthContext(
                user_id=current_user.id,
                age_group=AgeGroup.ADULT.value,
                activity_level=ActivityLevel.MODERATELY_ACTIVE.value
            )
            db.add(context)
        
        # Update fields
        update_data = context_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field in ["age_group", "activity_level"] and value:
                setattr(context, field, value.value)
            else:
                setattr(context, field, value)
        
        # Calculate BMI if height and weight provided
        if context.height_cm and context.weight_kg:
            height_m = context.height_cm / 100
            context.bmi = round(context.weight_kg / (height_m ** 2), 1)
        
        context.updated_at = datetime.utcnow()
        await db.commit()
        
        # Invalidate user cache
        await cache_manager.invalidate_user_cache(str(current_user.id))
        
        logger.info(f"Health context updated for user {current_user.id}")
        
        return JSONResponse(content={"message": "Health context updated successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating health context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update health context"
        )


@router.get("/recommendations")
async def get_health_recommendations(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get personalized health recommendations based on user profiles."""
    try:
        # Get user's health profiles
        result = await db.execute(
            select(HealthProfile).where(HealthProfile.user_id == current_user.id)
        )
        profiles = result.scalars().all()
        
        recommendations = []
        
        # Generate recommendations based on profiles
        for profile in profiles:
            profile_recommendations = _generate_profile_recommendations(profile)
            recommendations.extend(profile_recommendations)
        
        # Add general recommendations if no profiles
        if not profiles:
            recommendations = [
                "Consider setting up your health profiles for personalized recommendations",
                "Maintain a balanced diet with variety of nutrients",
                "Stay hydrated and limit processed foods",
                "Regular physical activity supports overall health"
            ]
        
        return JSONResponse(content={
            "recommendations": recommendations[:10],  # Limit to 10
            "profile_count": len(profiles)
        })
        
    except Exception as e:
        logger.error(f"Error generating health recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


def _generate_profile_recommendations(profile: HealthProfile) -> List[str]:
    """Generate recommendations based on health profile."""
    recommendations = []
    
    if profile.profile_type == "diabetes":
        recommendations.extend([
            "Monitor carbohydrate intake and choose complex carbs",
            "Include fiber-rich foods to help stabilize blood sugar",
            "Limit added sugars and processed foods"
        ])
        
        if profile.severity == "severe":
            recommendations.append("Consult your healthcare provider before making dietary changes")
    
    elif profile.profile_type == "hypertension":
        recommendations.extend([
            "Reduce sodium intake to less than 2300mg per day",
            "Include potassium-rich foods like bananas and leafy greens",
            "Follow DASH diet principles for optimal blood pressure control"
        ])
    
    elif profile.profile_type == "heart_disease":
        recommendations.extend([
            "Choose heart-healthy fats like olive oil and nuts",
            "Limit saturated and trans fats",
            "Include omega-3 rich foods like fish"
        ])
    
    elif profile.profile_type == "kidney_disease":
        recommendations.extend([
            "Monitor protein intake as recommended by your doctor",
            "Limit phosphorus and potassium if advised",
            "Control fluid intake if necessary"
        ])
    
    elif profile.profile_type == "obesity":
        recommendations.extend([
            "Focus on portion control and calorie balance",
            "Choose nutrient-dense, lower-calorie foods",
            "Increase physical activity gradually"
        ])
    
    return recommendations
