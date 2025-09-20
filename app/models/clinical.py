"""
Enhanced clinical models for diabetes management
Based on ADA/EASD guidelines and clinical best practices
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class DiabetesType(str, Enum):
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"
    PREDIABETES = "prediabetes"

class DiabetesSeverity(str, Enum):
    WELL_CONTROLLED = "well_controlled"  # HbA1c < 7%
    MODERATELY_CONTROLLED = "moderately_controlled"  # HbA1c 7-8%
    POORLY_CONTROLLED = "poorly_controlled"  # HbA1c > 8%

class DiabetesMedication(str, Enum):
    METFORMIN = "metformin"
    INSULIN_RAPID = "insulin_rapid"
    INSULIN_LONG = "insulin_long"
    SULFONYLUREAS = "sulfonylureas"
    GLP1_AGONISTS = "glp1_agonists"
    SGLT2_INHIBITORS = "sglt2_inhibitors"

class EnhancedDiabetesProfile(BaseModel):
    """Enhanced diabetes profile with clinical parameters"""
    diabetes_type: DiabetesType
    severity: DiabetesSeverity
    hba1c_target: float = Field(ge=5.0, le=12.0, description="Target HbA1c percentage")
    current_hba1c: Optional[float] = Field(None, ge=4.0, le=15.0)
    glucose_target_range: Dict[str, float] = Field(
        default={"fasting": 80, "post_meal": 140},
        description="Target glucose levels in mg/dL"
    )
    medications: List[DiabetesMedication] = []
    carb_ratio: Optional[float] = Field(None, description="Insulin-to-carb ratio for Type 1")
    meal_timing_sensitivity: bool = Field(True, description="Sensitive to meal timing")
    hypoglycemia_risk: bool = Field(False, description="History of hypoglycemic episodes")
    
class ClinicalRecommendationContext(BaseModel):
    """Context for generating clinically-informed recommendations"""
    evidence_level: str = Field(description="ADA/EASD guideline reference level")
    contraindications: List[str] = []
    drug_interactions: List[str] = []
    timing_considerations: Dict[str, str] = {}
    emergency_considerations: Optional[str] = None

class ClinicalAnalysisResult(BaseModel):
    """Result of clinical analysis with evidence-based recommendations"""
    ada_compliance: bool
    risk_level: str
    evidence_level: str
    clinical_reasoning: str
    enhanced_recommendations: List[str]
    medication_considerations: Dict[str, str]
    clinical_alternatives: List[Dict[str, Any]]
    timing_guidance: Dict[str, str]
    emergency_considerations: Dict[str, str]
    confidence_metrics: Dict[str, float]