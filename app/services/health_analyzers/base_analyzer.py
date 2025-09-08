"""
Base health analyzer with common functionality for all health conditions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import structlog

from app.models.health import NutritionData, UserHealthContext, ProfileType, Severity

logger = structlog.get_logger(__name__)


@dataclass
class AnalysisResult:
    """Result from health analysis."""
    score: int  # 0-100
    safety_level: str  # safe, caution, warning, danger
    recommendations: List[str]
    warnings: List[Dict[str, Any]]
    detailed_scores: Dict[str, int]
    reasoning: str


class BaseHealthAnalyzer(ABC):
    """Base class for all health condition analyzers."""
    
    def __init__(self, profile_type: ProfileType):
        self.profile_type = profile_type
        self.logger = structlog.get_logger(f"{__name__}.{profile_type.value}")
    
    @abstractmethod
    async def analyze(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        severity: Severity
    ) -> AnalysisResult:
        """
        Analyze nutrition data for specific health condition.
        
        Args:
            nutrition_data: Extracted nutrition information
            user_context: User's complete health context
            severity: Condition severity level
            
        Returns:
            Analysis result with scoring and recommendations
        """
        pass
    
    def _calculate_weighted_score(
        self,
        component_scores: Dict[str, Tuple[int, float]]
    ) -> int:
        """
        Calculate weighted average score from components.
        
        Args:
            component_scores: Dict of {component: (score, weight)}
            
        Returns:
            Weighted average score (0-100)
        """
        total_weighted_score = 0
        total_weight = 0
        
        for component, (score, weight) in component_scores.items():
            total_weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50  # Default neutral score
        
        return min(100, max(0, int(total_weighted_score / total_weight)))
    
    def _get_safety_level(self, score: int, warnings: List[Dict[str, Any]]) -> str:
        """Determine safety level based on score and warnings."""
        critical_warnings = [w for w in warnings if w.get("severity") == "critical"]
        high_warnings = [w for w in warnings if w.get("severity") == "high"]
        
        if critical_warnings or score < 20:
            return "danger"
        elif high_warnings or score < 40:
            return "warning"
        elif score < 60:
            return "caution"
        else:
            return "safe"
    
    def _check_allergen_safety(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext
    ) -> List[Dict[str, Any]]:
        """Check for allergen conflicts."""
        warnings = []
        
        product_allergens = set(allergen.lower() for allergen in nutrition_data.allergens)
        user_allergies = set(allergy.lower() for allergy in user_context.allergies)
        
        conflicts = product_allergens.intersection(user_allergies)
        
        for allergen in conflicts:
            warnings.append({
                "type": "allergen_conflict",
                "severity": "critical",
                "message": f"Contains {allergen.title()}, which you're allergic to",
                "allergen": allergen
            })
        
        return warnings
    
    def _check_dietary_restrictions(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext
    ) -> List[str]:
        """Check dietary restriction compliance."""
        recommendations = []
        
        # Check common restrictions
        restrictions = []
        for profile in user_context.primary_profiles:
            restrictions.extend(profile.restrictions)
        
        if "vegetarian" in user_context.dietary_preferences:
            # Check for meat ingredients (simplified)
            meat_indicators = ["beef", "pork", "chicken", "turkey", "fish", "meat"]
            ingredients_text = " ".join(nutrition_data.ingredients).lower()
            
            for meat in meat_indicators:
                if meat in ingredients_text:
                    recommendations.append(f"Contains {meat} - not suitable for vegetarian diet")
                    break
        
        if "vegan" in user_context.dietary_preferences:
            # Check for animal products
            animal_products = ["milk", "egg", "honey", "gelatin", "whey", "casein"]
            ingredients_text = " ".join(nutrition_data.ingredients).lower()
            
            for product in animal_products:
                if product in ingredients_text:
                    recommendations.append(f"Contains {product} - not suitable for vegan diet")
                    break
        
        return recommendations
    
    def _calculate_glycemic_impact(self, nutrition_data: NutritionData) -> int:
        """Calculate estimated glycemic impact (0-100)."""
        # Simplified glycemic impact calculation
        total_carbs = nutrition_data.carbohydrates
        fiber = nutrition_data.fiber or 0
        sugar = nutrition_data.sugar or 0
        
        # Net carbs
        net_carbs = max(0, total_carbs - fiber)
        
        # Higher sugar content increases glycemic impact
        sugar_factor = min(sugar / total_carbs if total_carbs > 0 else 0, 1.0)
        
        # Base impact from net carbs (assuming per serving)
        base_impact = min(net_carbs * 2, 100)  # Scale to 0-100
        
        # Adjust for sugar content
        glycemic_impact = base_impact * (1 + sugar_factor * 0.5)
        
        return min(100, int(glycemic_impact))
    
    def _assess_processing_level(self, nutrition_data: NutritionData) -> Tuple[str, int]:
        """Assess food processing level using NOVA classification."""
        ingredients_count = len(nutrition_data.ingredients)
        additives_count = len(nutrition_data.additives)
        
        # Simple NOVA-like classification
        if ingredients_count <= 1 and additives_count == 0:
            return "unprocessed", 90
        elif ingredients_count <= 5 and additives_count <= 2:
            return "minimally_processed", 70
        elif additives_count <= 5:
            return "processed", 50
        else:
            return "ultra_processed", 20
    
    def _generate_base_recommendations(
        self,
        nutrition_data: NutritionData,
        score: int
    ) -> List[str]:
        """Generate base recommendations applicable to all conditions."""
        recommendations = []
        
        if score < 40:
            recommendations.append("Consider finding a healthier alternative to this product")
        
        # Check sodium levels
        sodium = nutrition_data.sodium or 0
        if sodium > 600:  # High sodium per serving
            recommendations.append("This product is high in sodium - consider lower-sodium alternatives")
        
        # Check sugar levels
        sugar = nutrition_data.sugar or 0
        if sugar > 15:  # High sugar per serving
            recommendations.append("This product is high in sugar - consume in moderation")
        
        # Check fiber
        fiber = nutrition_data.fiber or 0
        if fiber < 3 and nutrition_data.carbohydrates > 15:
            recommendations.append("Look for higher-fiber alternatives for better digestive health")
        
        return recommendations
