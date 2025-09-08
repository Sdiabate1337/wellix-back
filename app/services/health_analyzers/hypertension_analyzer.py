"""
Hypertension-specific health analyzer with sodium and heart health focus.
"""

from typing import Dict, Any, List, Tuple
import structlog

from app.models.health import NutritionData, UserHealthContext, ProfileType, Severity
from app.services.health_analyzers.base_analyzer import BaseHealthAnalyzer, AnalysisResult

logger = structlog.get_logger(__name__)


class HypertensionAnalyzer(BaseHealthAnalyzer):
    """Specialized analyzer for hypertension management."""
    
    def __init__(self):
        super().__init__(ProfileType.HYPERTENSION)
    
    async def analyze(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        severity: Severity
    ) -> AnalysisResult:
        """Analyze nutrition data for hypertension management."""
        
        # Component scoring
        sodium_score, sodium_warnings = self._analyze_sodium(nutrition_data, severity)
        potassium_score = self._analyze_potassium(nutrition_data)
        fat_score, fat_warnings = self._analyze_fats(nutrition_data, severity)
        fiber_score = self._analyze_fiber(nutrition_data)
        processing_score = self._analyze_processing_level(nutrition_data)
        
        # Calculate weighted overall score
        component_scores = {
            "sodium": (sodium_score, 0.35),
            "potassium": (potassium_score, 0.2),
            "fats": (fat_score, 0.2),
            "fiber": (fiber_score, 0.15),
            "processing": (processing_score, 0.1)
        }
        
        overall_score = self._calculate_weighted_score(component_scores)
        
        # Collect all warnings
        all_warnings = sodium_warnings + fat_warnings
        all_warnings.extend(self._check_allergen_safety(nutrition_data, user_context))
        
        # Generate recommendations
        recommendations = self._generate_hypertension_recommendations(
            nutrition_data, overall_score, severity
        )
        recommendations.extend(self._check_dietary_restrictions(nutrition_data, user_context))
        
        # Detailed component scores
        detailed_scores = {
            "sodium_control": sodium_score,
            "potassium_balance": potassium_score,
            "heart_healthy_fats": fat_score,
            "fiber_content": fiber_score,
            "processing_level": processing_score
        }
        
        # Generate reasoning
        reasoning = self._generate_reasoning(nutrition_data, detailed_scores, severity)
        
        return AnalysisResult(
            score=overall_score,
            safety_level=self._get_safety_level(overall_score, all_warnings),
            recommendations=recommendations,
            warnings=all_warnings,
            detailed_scores=detailed_scores,
            reasoning=reasoning
        )
    
    def _analyze_sodium(
        self,
        nutrition_data: NutritionData,
        severity: Severity
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Analyze sodium content for hypertension management."""
        sodium = nutrition_data.sodium or 0
        
        warnings = []
        
        # Severity-based sodium thresholds (mg per serving)
        if severity == Severity.SEVERE:
            critical_threshold = 300
            high_threshold = 200
            moderate_threshold = 100
        elif severity == Severity.MODERATE:
            critical_threshold = 500
            high_threshold = 350
            moderate_threshold = 200
        else:  # MILD
            critical_threshold = 700
            high_threshold = 500
            moderate_threshold = 300
        
        if sodium > critical_threshold:
            score = 10
            warnings.append({
                "type": "critical_sodium",
                "severity": "critical",
                "message": f"Very high sodium content ({sodium}mg) - avoid this product",
                "value": sodium,
                "threshold": critical_threshold
            })
        elif sodium > high_threshold:
            score = 25
            warnings.append({
                "type": "high_sodium",
                "severity": "high",
                "message": f"High sodium content ({sodium}mg) - consume very sparingly",
                "value": sodium,
                "threshold": high_threshold
            })
        elif sodium > moderate_threshold:
            score = 50
            warnings.append({
                "type": "moderate_sodium",
                "severity": "medium",
                "message": f"Moderate sodium content ({sodium}mg) - limit portion size",
                "value": sodium,
                "threshold": moderate_threshold
            })
        else:
            score = 90
        
        return score, warnings
    
    def _analyze_potassium(self, nutrition_data: NutritionData) -> int:
        """Analyze potassium content (beneficial for blood pressure)."""
        potassium = nutrition_data.potassium or 0
        
        # Potassium is beneficial for hypertension
        if potassium >= 400:  # Excellent source
            return 95
        elif potassium >= 200:  # Good source
            return 80
        elif potassium >= 100:  # Some potassium
            return 60
        else:  # Low/no potassium
            return 40
    
    def _analyze_fats(
        self,
        nutrition_data: NutritionData,
        severity: Severity
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Analyze fat content for heart health."""
        total_fat = nutrition_data.total_fat
        saturated_fat = nutrition_data.saturated_fat or 0
        trans_fat = nutrition_data.trans_fat or 0
        
        warnings = []
        score = 80  # Start with good score
        
        # Trans fat is critical
        if trans_fat > 0:
            score = min(score, 20)
            warnings.append({
                "type": "trans_fat",
                "severity": "critical",
                "message": f"Contains {trans_fat}g trans fat - avoid completely",
                "value": trans_fat
            })
        
        # Saturated fat thresholds
        if severity == Severity.SEVERE:
            sat_fat_threshold = 3
        elif severity == Severity.MODERATE:
            sat_fat_threshold = 5
        else:  # MILD
            sat_fat_threshold = 7
        
        if saturated_fat > sat_fat_threshold:
            score = min(score, 40)
            warnings.append({
                "type": "high_saturated_fat",
                "severity": "medium",
                "message": f"High saturated fat ({saturated_fat}g) - limit consumption",
                "value": saturated_fat,
                "threshold": sat_fat_threshold
            })
        
        # Total fat percentage of calories
        if total_fat > 0 and nutrition_data.calories > 0:
            fat_calories = total_fat * 9
            fat_percentage = (fat_calories / nutrition_data.calories) * 100
            
            if fat_percentage > 40:  # More than 40% calories from fat
                score = min(score, 50)
        
        return score, warnings
    
    def _analyze_fiber(self, nutrition_data: NutritionData) -> int:
        """Analyze fiber content (beneficial for heart health)."""
        fiber = nutrition_data.fiber or 0
        
        if fiber >= 5:  # High fiber
            return 90
        elif fiber >= 3:  # Good fiber
            return 75
        elif fiber >= 1:  # Some fiber
            return 60
        else:  # No fiber
            return 40
    
    def _analyze_processing_level(self, nutrition_data: NutritionData) -> int:
        """Analyze processing level (less processed is better for hypertension)."""
        processing_level, score = self._assess_processing_level(nutrition_data)
        
        # Ultra-processed foods often high in sodium
        if processing_level == "ultra_processed":
            return max(score - 20, 10)
        
        return score
    
    def _generate_hypertension_recommendations(
        self,
        nutrition_data: NutritionData,
        score: int,
        severity: Severity
    ) -> List[str]:
        """Generate hypertension-specific recommendations."""
        recommendations = []
        
        # Base recommendations
        recommendations.extend(self._generate_base_recommendations(nutrition_data, score))
        
        # Hypertension-specific recommendations
        sodium = nutrition_data.sodium or 0
        
        if sodium > 400:
            recommendations.append("Choose low-sodium alternatives to help manage blood pressure")
        
        if (nutrition_data.potassium or 0) > 200:
            recommendations.append("Good potassium content helps counteract sodium effects")
        
        if (nutrition_data.trans_fat or 0) > 0:
            recommendations.append("Avoid trans fats completely - they increase cardiovascular risk")
        
        if (nutrition_data.saturated_fat or 0) > 5:
            recommendations.append("Limit saturated fat intake for better heart health")
        
        if (nutrition_data.fiber or 0) >= 3:
            recommendations.append("Good fiber content supports heart health")
        
        # DASH diet recommendations
        recommendations.append("Consider following DASH diet principles for optimal blood pressure control")
        
        if severity == Severity.SEVERE and score < 50:
            recommendations.append("Consult your cardiologist before including this in your diet")
        
        return recommendations
    
    def _generate_reasoning(
        self,
        nutrition_data: NutritionData,
        detailed_scores: Dict[str, int],
        severity: Severity
    ) -> str:
        """Generate detailed reasoning for the analysis."""
        sodium = nutrition_data.sodium or 0
        potassium = nutrition_data.potassium or 0
        saturated_fat = nutrition_data.saturated_fat or 0
        
        reasoning_parts = [
            f"Hypertension Analysis for {severity.value} condition:",
            f"• Sodium content: {sodium}mg (score: {detailed_scores['sodium_control']})",
            f"• Potassium content: {potassium}mg (score: {detailed_scores['potassium_balance']})",
            f"• Saturated fat: {saturated_fat}g (score: {detailed_scores['heart_healthy_fats']})"
        ]
        
        if detailed_scores['sodium_control'] < 50:
            reasoning_parts.append("• High sodium content may elevate blood pressure")
        
        if detailed_scores['potassium_balance'] > 70:
            reasoning_parts.append("• Good potassium content helps regulate blood pressure")
        
        if detailed_scores['heart_healthy_fats'] < 50:
            reasoning_parts.append("• Fat profile may negatively impact cardiovascular health")
        
        return "\n".join(reasoning_parts)
