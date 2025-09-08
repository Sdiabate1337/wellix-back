"""
Health analyzer factory for creating condition-specific analyzers.
"""

from typing import Dict, List, Optional
import structlog

from app.models.health import ProfileType, UserHealthContext, NutritionData, Severity
from app.services.health_analyzers.base_analyzer import BaseHealthAnalyzer, AnalysisResult
from app.services.health_analyzers.diabetes_analyzer import DiabetesAnalyzer
from app.services.health_analyzers.hypertension_analyzer import HypertensionAnalyzer

logger = structlog.get_logger(__name__)


class GeneralAnalyzer(BaseHealthAnalyzer):
    """General health analyzer for users without specific conditions."""
    
    def __init__(self):
        super().__init__(ProfileType.GENERAL)
    
    async def analyze(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        severity: Severity
    ) -> AnalysisResult:
        """Analyze nutrition data for general health."""
        
        # Component scoring for general health
        nutrition_balance_score = self._analyze_nutrition_balance(nutrition_data)
        processing_score = self._analyze_processing_level(nutrition_data)
        sodium_score = self._analyze_general_sodium(nutrition_data)
        sugar_score = self._analyze_general_sugar(nutrition_data)
        fiber_score = self._analyze_general_fiber(nutrition_data)
        
        # Calculate weighted overall score
        component_scores = {
            "nutrition_balance": (nutrition_balance_score, 0.25),
            "processing_level": (processing_score, 0.2),
            "sodium_levels": (sodium_score, 0.2),
            "sugar_content": (sugar_score, 0.2),
            "fiber_content": (fiber_score, 0.15)
        }
        
        overall_score = self._calculate_weighted_score(component_scores)
        
        # Check for warnings
        warnings = self._check_allergen_safety(nutrition_data, user_context)
        
        # Generate recommendations
        recommendations = self._generate_general_recommendations(nutrition_data, overall_score)
        recommendations.extend(self._check_dietary_restrictions(nutrition_data, user_context))
        
        # Detailed scores
        detailed_scores = {
            "nutrition_balance": nutrition_balance_score,
            "processing_level": processing_score,
            "sodium_levels": sodium_score,
            "sugar_content": sugar_score,
            "fiber_content": fiber_score
        }
        
        reasoning = self._generate_general_reasoning(nutrition_data, detailed_scores)
        
        return AnalysisResult(
            score=overall_score,
            safety_level=self._get_safety_level(overall_score, warnings),
            recommendations=recommendations,
            warnings=warnings,
            detailed_scores=detailed_scores,
            reasoning=reasoning
        )
    
    def _analyze_nutrition_balance(self, nutrition_data: NutritionData) -> int:
        """Analyze overall nutritional balance."""
        macros = nutrition_data.get_macronutrient_percentages()
        
        protein_pct = macros["protein"]
        carb_pct = macros["carbohydrates"]
        fat_pct = macros["fat"]
        
        score = 80  # Start with good score
        
        # Ideal ranges: Protein 10-35%, Carbs 45-65%, Fat 20-35%
        if protein_pct < 5:
            score -= 15  # Very low protein
        elif protein_pct > 40:
            score -= 10  # Very high protein
        
        if carb_pct > 70:
            score -= 15  # Very high carbs
        elif carb_pct < 30:
            score -= 10  # Very low carbs
        
        if fat_pct > 40:
            score -= 15  # Very high fat
        elif fat_pct < 15:
            score -= 10  # Very low fat
        
        return max(score, 10)
    
    def _analyze_processing_level(self, nutrition_data: NutritionData) -> int:
        """Analyze food processing level."""
        _, score = self._assess_processing_level(nutrition_data)
        return score
    
    def _analyze_general_sodium(self, nutrition_data: NutritionData) -> int:
        """Analyze sodium for general population."""
        sodium = nutrition_data.sodium or 0
        
        if sodium > 800:  # Very high
            return 20
        elif sodium > 500:  # High
            return 50
        elif sodium > 300:  # Moderate
            return 70
        else:  # Low
            return 90
    
    def _analyze_general_sugar(self, nutrition_data: NutritionData) -> int:
        """Analyze sugar content for general population."""
        sugar = nutrition_data.sugar or 0
        
        if sugar > 20:  # Very high
            return 25
        elif sugar > 12:  # High
            return 50
        elif sugar > 6:  # Moderate
            return 75
        else:  # Low
            return 90
    
    def _analyze_general_fiber(self, nutrition_data: NutritionData) -> int:
        """Analyze fiber content for general health."""
        fiber = nutrition_data.fiber or 0
        
        if fiber >= 5:  # High fiber
            return 95
        elif fiber >= 3:  # Good fiber
            return 80
        elif fiber >= 1:  # Some fiber
            return 60
        else:  # No fiber
            return 40
    
    def _generate_general_recommendations(self, nutrition_data: NutritionData, score: int) -> List[str]:
        """Generate general health recommendations."""
        recommendations = self._generate_base_recommendations(nutrition_data, score)
        
        # Add general health tips
        if (nutrition_data.fiber or 0) >= 3:
            recommendations.append("Good fiber content supports digestive health")
        
        macros = nutrition_data.get_macronutrient_percentages()
        if macros["protein"] >= 15:
            recommendations.append("Good protein content for muscle maintenance")
        
        return recommendations
    
    def _generate_general_reasoning(self, nutrition_data: NutritionData, detailed_scores: Dict[str, int]) -> str:
        """Generate reasoning for general health analysis."""
        return f"""General Health Analysis:
• Nutritional balance: {detailed_scores['nutrition_balance']}/100
• Processing level: {detailed_scores['processing_level']}/100
• Sodium content: {detailed_scores['sodium_levels']}/100
• Sugar content: {detailed_scores['sugar_content']}/100
• Fiber content: {detailed_scores['fiber_content']}/100

Overall assessment based on general nutrition guidelines for healthy adults."""


class HealthAnalyzerFactory:
    """Factory for creating health condition analyzers."""
    
    _analyzers: Dict[ProfileType, BaseHealthAnalyzer] = {}
    
    @classmethod
    def get_analyzer(cls, profile_type: ProfileType) -> BaseHealthAnalyzer:
        """Get analyzer for specific health profile type."""
        if profile_type not in cls._analyzers:
            cls._analyzers[profile_type] = cls._create_analyzer(profile_type)
        
        return cls._analyzers[profile_type]
    
    @classmethod
    def _create_analyzer(cls, profile_type: ProfileType) -> BaseHealthAnalyzer:
        """Create analyzer instance for profile type."""
        analyzer_map = {
            ProfileType.DIABETES: DiabetesAnalyzer,
            ProfileType.HYPERTENSION: HypertensionAnalyzer,
            ProfileType.GENERAL: GeneralAnalyzer,
            # Add more analyzers as implemented
            ProfileType.HEART_DISEASE: GeneralAnalyzer,  # Placeholder
            ProfileType.KIDNEY_DISEASE: GeneralAnalyzer,  # Placeholder
            ProfileType.OBESITY: GeneralAnalyzer,  # Placeholder
        }
        
        analyzer_class = analyzer_map.get(profile_type, GeneralAnalyzer)
        return analyzer_class()
    
    @classmethod
    async def analyze_for_user(
        cls,
        nutrition_data: NutritionData,
        user_context: UserHealthContext
    ) -> Dict[str, AnalysisResult]:
        """
        Analyze nutrition data for all user's health profiles.
        
        Returns:
            Dictionary mapping profile type to analysis result
        """
        results = {}
        
        # If no profiles, use general analyzer
        if not user_context.primary_profiles:
            analyzer = cls.get_analyzer(ProfileType.GENERAL)
            result = await analyzer.analyze(nutrition_data, user_context, Severity.MILD)
            results[ProfileType.GENERAL.value] = result
            return results
        
        # Analyze for each profile
        for profile in user_context.primary_profiles:
            analyzer = cls.get_analyzer(profile.profile_type)
            result = await analyzer.analyze(nutrition_data, user_context, profile.severity)
            results[profile.profile_type.value] = result
        
        return results
    
    @classmethod
    def get_overall_score(cls, profile_results: Dict[str, AnalysisResult]) -> int:
        """Calculate overall score from multiple profile analyses."""
        if not profile_results:
            return 50
        
        # Weight primary conditions more heavily
        total_weighted_score = 0
        total_weight = 0
        
        for profile_type, result in profile_results.items():
            # Primary conditions get higher weight
            weight = 1.0 if profile_type == ProfileType.GENERAL.value else 1.5
            total_weighted_score += result.score * weight
            total_weight += weight
        
        return int(total_weighted_score / total_weight) if total_weight > 0 else 50
