"""
Diabetes-specific health analyzer with glycemic control focus.
"""

from typing import Dict, Any, List, Tuple
import structlog

from app.models.health import NutritionData, UserHealthContext, ProfileType, Severity
from app.services.health_analyzers.base_analyzer import BaseHealthAnalyzer, AnalysisResult

logger = structlog.get_logger(__name__)


class DiabetesAnalyzer(BaseHealthAnalyzer):
    """Specialized analyzer for diabetes management."""
    
    def __init__(self):
        super().__init__(ProfileType.DIABETES)
    
    async def analyze(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        severity: Severity
    ) -> AnalysisResult:
        """Analyze nutrition data for diabetes management."""
        
        # Component scoring
        carb_score, carb_warnings = self._analyze_carbohydrates(nutrition_data, severity)
        sugar_score, sugar_warnings = self._analyze_sugars(nutrition_data, severity)
        fiber_score = self._analyze_fiber(nutrition_data)
        glycemic_score = self._analyze_glycemic_impact(nutrition_data)
        sodium_score = self._analyze_sodium(nutrition_data, severity)
        
        # Calculate weighted overall score
        component_scores = {
            "carbohydrates": (carb_score, 0.3),
            "sugars": (sugar_score, 0.25),
            "fiber": (fiber_score, 0.2),
            "glycemic_impact": (glycemic_score, 0.15),
            "sodium": (sodium_score, 0.1)
        }
        
        overall_score = self._calculate_weighted_score(component_scores)
        
        # Collect all warnings
        all_warnings = carb_warnings + sugar_warnings
        all_warnings.extend(self._check_allergen_safety(nutrition_data, user_context))
        
        # Generate recommendations
        recommendations = self._generate_diabetes_recommendations(
            nutrition_data, overall_score, severity
        )
        recommendations.extend(self._check_dietary_restrictions(nutrition_data, user_context))
        
        # Detailed component scores
        detailed_scores = {
            "carbohydrate_management": carb_score,
            "sugar_control": sugar_score,
            "fiber_content": fiber_score,
            "glycemic_impact": glycemic_score,
            "sodium_levels": sodium_score
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
    
    def _analyze_carbohydrates(
        self,
        nutrition_data: NutritionData,
        severity: Severity
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Analyze carbohydrate content for diabetes management."""
        total_carbs = nutrition_data.carbohydrates
        fiber = nutrition_data.fiber or 0
        net_carbs = max(0, total_carbs - fiber)
        
        warnings = []
        
        # Severity-based thresholds
        if severity == Severity.SEVERE:
            high_threshold = 15
            moderate_threshold = 10
        elif severity == Severity.MODERATE:
            high_threshold = 25
            moderate_threshold = 15
        else:  # MILD
            high_threshold = 35
            moderate_threshold = 20
        
        if net_carbs > high_threshold:
            score = 20
            warnings.append({
                "type": "high_carbohydrates",
                "severity": "high",
                "message": f"High net carbohydrates ({net_carbs}g) - may cause blood sugar spike",
                "value": net_carbs,
                "threshold": high_threshold
            })
        elif net_carbs > moderate_threshold:
            score = 50
            warnings.append({
                "type": "moderate_carbohydrates",
                "severity": "medium",
                "message": f"Moderate carbohydrates ({net_carbs}g) - monitor blood sugar",
                "value": net_carbs,
                "threshold": moderate_threshold
            })
        else:
            score = 85
        
        return score, warnings
    
    def _analyze_sugars(
        self,
        nutrition_data: NutritionData,
        severity: Severity
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Analyze sugar content for diabetes management."""
        total_sugar = nutrition_data.sugar or 0
        added_sugar = nutrition_data.added_sugar or 0
        
        warnings = []
        
        # Severity-based sugar thresholds
        if severity == Severity.SEVERE:
            high_sugar_threshold = 5
            moderate_sugar_threshold = 3
        elif severity == Severity.MODERATE:
            high_sugar_threshold = 10
            moderate_sugar_threshold = 6
        else:  # MILD
            high_sugar_threshold = 15
            moderate_sugar_threshold = 10
        
        if total_sugar > high_sugar_threshold:
            score = 15
            warnings.append({
                "type": "high_sugar",
                "severity": "critical" if severity == Severity.SEVERE else "high",
                "message": f"High sugar content ({total_sugar}g) - avoid or consume very sparingly",
                "value": total_sugar,
                "threshold": high_sugar_threshold
            })
        elif total_sugar > moderate_sugar_threshold:
            score = 40
            warnings.append({
                "type": "moderate_sugar",
                "severity": "medium",
                "message": f"Moderate sugar content ({total_sugar}g) - consume in small portions",
                "value": total_sugar,
                "threshold": moderate_sugar_threshold
            })
        else:
            score = 90
        
        # Additional penalty for added sugars
        if added_sugar > 0:
            score = max(score - 20, 10)
            warnings.append({
                "type": "added_sugar",
                "severity": "medium",
                "message": f"Contains {added_sugar}g added sugar - prefer naturally sweetened alternatives",
                "value": added_sugar
            })
        
        return score, warnings
    
    def _analyze_fiber(self, nutrition_data: NutritionData) -> int:
        """Analyze fiber content (beneficial for blood sugar control)."""
        fiber = nutrition_data.fiber or 0
        total_carbs = nutrition_data.carbohydrates
        
        if total_carbs == 0:
            return 80  # No carbs, fiber not relevant
        
        fiber_ratio = fiber / total_carbs
        
        if fiber_ratio >= 0.25:  # 25% or more fiber
            return 95
        elif fiber_ratio >= 0.15:  # 15-25% fiber
            return 80
        elif fiber_ratio >= 0.08:  # 8-15% fiber
            return 60
        else:  # Less than 8% fiber
            return 30
    
    def _analyze_glycemic_impact(self, nutrition_data: NutritionData) -> int:
        """Analyze estimated glycemic impact."""
        glycemic_impact = self._calculate_glycemic_impact(nutrition_data)
        
        # Invert score (lower glycemic impact = higher score)
        if glycemic_impact <= 20:
            return 95
        elif glycemic_impact <= 40:
            return 75
        elif glycemic_impact <= 60:
            return 50
        else:
            return 25
    
    def _analyze_sodium(self, nutrition_data: NutritionData, severity: Severity) -> int:
        """Analyze sodium content (important for diabetics with hypertension risk)."""
        sodium = nutrition_data.sodium or 0
        
        # Diabetics often have higher hypertension risk
        if severity == Severity.SEVERE:
            high_threshold = 400
            moderate_threshold = 250
        else:
            high_threshold = 600
            moderate_threshold = 400
        
        if sodium > high_threshold:
            return 25
        elif sodium > moderate_threshold:
            return 60
        else:
            return 85
    
    def _generate_diabetes_recommendations(
        self,
        nutrition_data: NutritionData,
        score: int,
        severity: Severity
    ) -> List[str]:
        """Generate diabetes-specific recommendations."""
        recommendations = []
        
        # Base recommendations
        recommendations.extend(self._generate_base_recommendations(nutrition_data, score))
        
        # Diabetes-specific recommendations
        net_carbs = max(0, nutrition_data.carbohydrates - (nutrition_data.fiber or 0))
        
        if net_carbs > 20:
            recommendations.append("Consider pairing with protein or healthy fats to slow glucose absorption")
        
        if (nutrition_data.sugar or 0) > 10:
            recommendations.append("Monitor blood glucose closely after consuming this product")
        
        if (nutrition_data.fiber or 0) < 3 and nutrition_data.carbohydrates > 10:
            recommendations.append("Look for higher-fiber alternatives to help stabilize blood sugar")
        
        if severity == Severity.SEVERE and score < 60:
            recommendations.append("Consult with your healthcare provider before including this in your meal plan")
        
        # Portion control recommendations
        if net_carbs > 15:
            recommendations.append("Consider consuming smaller portions to minimize blood sugar impact")
        
        return recommendations
    
    def _generate_reasoning(
        self,
        nutrition_data: NutritionData,
        detailed_scores: Dict[str, int],
        severity: Severity
    ) -> str:
        """Generate detailed reasoning for the analysis."""
        net_carbs = max(0, nutrition_data.carbohydrates - (nutrition_data.fiber or 0))
        sugar = nutrition_data.sugar or 0
        fiber = nutrition_data.fiber or 0
        
        reasoning_parts = [
            f"Diabetes Analysis for {severity.value} condition:",
            f"• Net carbohydrates: {net_carbs}g (score: {detailed_scores['carbohydrate_management']})",
            f"• Sugar content: {sugar}g (score: {detailed_scores['sugar_control']})",
            f"• Fiber content: {fiber}g (score: {detailed_scores['fiber_content']})"
        ]
        
        if detailed_scores['carbohydrate_management'] < 50:
            reasoning_parts.append("• High carbohydrate content may cause blood glucose spikes")
        
        if detailed_scores['sugar_control'] < 50:
            reasoning_parts.append("• High sugar content requires careful portion control")
        
        if detailed_scores['fiber_content'] > 70:
            reasoning_parts.append("• Good fiber content helps slow glucose absorption")
        
        return "\n".join(reasoning_parts)
