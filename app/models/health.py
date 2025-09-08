"""
Health profile models for multi-condition analysis.
Supports diabetes, hypertension, heart disease, kidney disease, obesity, and general wellness.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ProfileType(str, Enum):
    """Supported health profile types."""
    DIABETES = "diabetes"
    HYPERTENSION = "hypertension"
    HEART_DISEASE = "heart_disease"
    KIDNEY_DISEASE = "kidney_disease"
    OBESITY = "obesity"
    GENERAL = "general"


class Severity(str, Enum):
    """Health condition severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ActivityLevel(str, Enum):
    """User activity levels for personalization."""
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"


class AgeGroup(str, Enum):
    """Age groups for nutritional recommendations."""
    CHILD = "child"          # 2-12 years
    TEEN = "teen"            # 13-17 years
    YOUNG_ADULT = "young_adult"  # 18-30 years
    ADULT = "adult"          # 31-50 years
    MIDDLE_AGED = "middle_aged"  # 51-65 years
    SENIOR = "senior"        # 65+ years


class HealthProfile(BaseModel):
    """Individual health profile for specific conditions."""
    
    profile_type: ProfileType
    severity: Severity
    restrictions: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    medications: Optional[List[str]] = Field(default_factory=list)
    is_primary: bool = Field(default=False)
    notes: Optional[str] = None
    
    # Condition-specific parameters
    target_values: Optional[Dict[str, float]] = Field(default_factory=dict)
    
    @validator("restrictions")
    def validate_restrictions(cls, v):
        """Validate dietary restrictions."""
        allowed_restrictions = [
            "low_sodium", "low_sugar", "low_carb", "low_fat", "low_cholesterol",
            "high_fiber", "gluten_free", "dairy_free", "vegetarian", "vegan",
            "keto", "paleo", "mediterranean", "dash", "low_potassium", "low_phosphorus"
        ]
        for restriction in v:
            if restriction not in allowed_restrictions:
                raise ValueError(f"Invalid restriction: {restriction}")
        return v
    
    @validator("goals")
    def validate_goals(cls, v):
        """Validate health goals."""
        allowed_goals = [
            "weight_loss", "weight_gain", "weight_maintenance", "blood_sugar_control",
            "blood_pressure_control", "cholesterol_management", "heart_health",
            "kidney_health", "digestive_health", "energy_boost", "muscle_gain",
            "bone_health", "immune_support", "inflammation_reduction"
        ]
        for goal in v:
            if goal not in allowed_goals:
                raise ValueError(f"Invalid goal: {goal}")
        return v


class UserHealthContext(BaseModel):
    """Complete user health context for personalized analysis."""
    
    user_id: str
    primary_profiles: List[HealthProfile] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    dietary_preferences: List[str] = Field(default_factory=list)
    age_group: AgeGroup
    activity_level: ActivityLevel
    weight_goals: Optional[str] = None
    
    # Additional context
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    
    # Preferences
    preferred_language: str = Field(default="en")
    analysis_depth: str = Field(default="standard")  # quick, standard, comprehensive
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("allergies")
    def validate_allergies(cls, v):
        """Validate common allergens."""
        common_allergens = [
            "milk", "eggs", "fish", "shellfish", "tree_nuts", "peanuts",
            "wheat", "soybeans", "sesame", "gluten", "sulfites", "corn"
        ]
        for allergen in v:
            if allergen.lower() not in common_allergens:
                # Allow custom allergens but warn
                pass
        return [allergen.lower() for allergen in v]
    
    @validator("dietary_preferences")
    def validate_dietary_preferences(cls, v):
        """Validate dietary preferences."""
        allowed_preferences = [
            "vegetarian", "vegan", "pescatarian", "flexitarian", "omnivore",
            "keto", "paleo", "mediterranean", "dash", "whole30", "raw_food",
            "organic_only", "local_only", "halal", "kosher"
        ]
        for pref in v:
            if pref not in allowed_preferences:
                raise ValueError(f"Invalid dietary preference: {pref}")
        return v
    
    def get_primary_conditions(self) -> List[ProfileType]:
        """Get list of primary health conditions."""
        return [profile.profile_type for profile in self.primary_profiles if profile.is_primary]
    
    def has_condition(self, condition: ProfileType) -> bool:
        """Check if user has specific health condition."""
        return any(profile.profile_type == condition for profile in self.primary_profiles)
    
    def get_condition_severity(self, condition: ProfileType) -> Optional[Severity]:
        """Get severity level for specific condition."""
        for profile in self.primary_profiles:
            if profile.profile_type == condition:
                return profile.severity
        return None
    
    def calculate_bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight are available."""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            bmi = self.weight_kg / (height_m ** 2)
            return round(bmi, 1)
        return None
    
    def get_bmi_category(self) -> Optional[str]:
        """Get BMI category classification."""
        bmi = self.calculate_bmi()
        if not bmi:
            return None
        
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"


class NutritionData(BaseModel):
    """Structured nutrition data from OCR or API."""
    
    product_name: str
    brand: Optional[str] = None
    barcode: Optional[str] = None
    serving_size: str
    
    # Macronutrients (per serving)
    calories: float
    protein: float  # grams
    carbohydrates: float  # grams
    total_fat: float  # grams
    saturated_fat: Optional[float] = None  # grams
    trans_fat: Optional[float] = None  # grams
    fiber: Optional[float] = None  # grams
    sugar: Optional[float] = None  # grams
    added_sugar: Optional[float] = None  # grams
    
    # Micronutrients (per serving)
    sodium: Optional[float] = None  # mg
    potassium: Optional[float] = None  # mg
    cholesterol: Optional[float] = None  # mg
    calcium: Optional[float] = None  # mg
    iron: Optional[float] = None  # mg
    vitamin_c: Optional[float] = None  # mg
    vitamin_d: Optional[float] = None  # IU
    
    # Additional data
    ingredients: List[str] = Field(default_factory=list)
    allergens: List[str] = Field(default_factory=list)
    additives: List[str] = Field(default_factory=list)
    
    # Metadata
    data_source: str = "manual"  # manual, ocr, openfoodfacts, barcode_lookup
    confidence_score: Optional[float] = None
    
    @validator("calories", "protein", "carbohydrates", "total_fat")
    def validate_positive_values(cls, v):
        """Ensure core nutrition values are positive."""
        if v < 0:
            raise ValueError("Nutrition values must be positive")
        return v
    
    def calculate_calories_from_macros(self) -> float:
        """Calculate calories from macronutrients (4-4-9 rule)."""
        protein_cal = self.protein * 4
        carb_cal = self.carbohydrates * 4
        fat_cal = self.total_fat * 9
        return protein_cal + carb_cal + fat_cal
    
    def get_macronutrient_percentages(self) -> Dict[str, float]:
        """Calculate macronutrient distribution as percentages."""
        total_cal = self.calories
        if total_cal == 0:
            return {"protein": 0, "carbohydrates": 0, "fat": 0}
        
        return {
            "protein": round((self.protein * 4 / total_cal) * 100, 1),
            "carbohydrates": round((self.carbohydrates * 4 / total_cal) * 100, 1),
            "fat": round((self.total_fat * 9 / total_cal) * 100, 1)
        }


class AnalysisResult(BaseModel):
    """Complete analysis result for a food product."""
    
    # Basic info
    analysis_id: str
    user_id: str
    nutrition_data: NutritionData
    
    # Overall scoring
    overall_score: int = Field(ge=0, le=100)
    safety_score: int = Field(ge=0, le=100)
    
    # Profile-specific scores
    profile_scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Insights and recommendations
    safety_alerts: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Additional analysis
    glycemic_impact: Optional[int] = None
    processing_level: Optional[str] = None
    ingredient_quality_score: Optional[int] = None
    
    # Metadata
    processing_time_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_score_category(self) -> str:
        """Get overall score category."""
        if self.overall_score >= 80:
            return "excellent"
        elif self.overall_score >= 60:
            return "good"
        elif self.overall_score >= 40:
            return "fair"
        else:
            return "poor"
    
    def has_safety_concerns(self) -> bool:
        """Check if product has safety concerns."""
        return len(self.safety_alerts) > 0 or self.safety_score < 70


class ProductRecommendation(BaseModel):
    """Product recommendation with reasoning."""
    
    product_name: str
    brand: str
    barcode: Optional[str] = None
    recommendation_score: float = Field(ge=0, le=100)
    reasoning: str
    health_benefits: List[str] = Field(default_factory=list)
    availability: Optional[str] = None
    price_comparison: Optional[str] = None
    
    # Nutrition comparison
    nutrition_improvements: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
