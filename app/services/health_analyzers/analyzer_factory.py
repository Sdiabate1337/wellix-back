"""
Enhanced Health Analyzer Factory with LLM Integration Points.
Combines clinical algorithms with LLM intelligence for optimal health analysis.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import structlog

from app.models.health import (
    NutritionData, UserHealthContext, AnalysisResult, 
    ProfileType, Severity
)
from app.services.health_analyzers.base_analyzer import BaseHealthAnalyzer
from app.services.health_analyzers.diabetes_analyzer import DiabetesAnalyzer
from app.services.health_analyzers.hypertension_analyzer import HypertensionAnalyzer
from app.services.ai.nutrition_analysis_ai import mock_enhance_nutrition_analysis as enhance_nutrition_analysis

logger = structlog.get_logger(__name__)


class LLMIntegrationLevel(Enum):
    """Public API for LLM integration levels exposed to users."""
    ALGORITHMIC_ONLY = "algorithmic_only"       # Pure clinical algorithms
    LLM_ENHANCED = "llm_enhanced"               # Clinical + light AI enhancement
    HYBRID_BALANCED = "hybrid_balanced"         # Balanced clinical + AI
    LLM_DOMINANT = "llm_dominant"              # AI-dominant approach


class ContextualInsightLevel(Enum):
    """Internal AI enhancement levels - automatically selected based on context."""
    CLINICAL_FOCUS = "clinical_focus"           # Minimal AI, clinical foundation
    SMART_ENHANCEMENT = "smart_enhancement"     # Light AI enhancement
    BALANCED_INTELLIGENCE = "balanced_intelligence"  # Balanced clinical + AI
    MAXIMUM_INSIGHT = "maximum_insight"         # AI-dominant for high-risk


class EnhancedAnalysisResult:
    """Enhanced analysis result with LLM insights."""
    
    def __init__(
        self,
        algorithmic_result: AnalysisResult,
        llm_insights: Optional[Dict[str, Any]] = None,
        hybrid_score: Optional[int] = None,
        confidence_metrics: Optional[Dict[str, float]] = None
    ):
        self.algorithmic_result = algorithmic_result
        self.llm_insights = llm_insights or {}
        self.hybrid_score = hybrid_score or algorithmic_result.score
        self.confidence_metrics = confidence_metrics or {}
        
        # Enhanced properties
        self.final_score = hybrid_score or algorithmic_result.score
        self.enhanced_recommendations = self._merge_recommendations()
        self.risk_assessment = self._enhanced_risk_assessment()
        self.personalized_insights = self._extract_personalized_insights()
    
    def _merge_recommendations(self) -> List[str]:
        """Merge algorithmic and LLM recommendations."""
        recommendations = list(self.algorithmic_result.recommendations)
        
        if self.llm_insights.get("additional_recommendations"):
            recommendations.extend(self.llm_insights["additional_recommendations"])
        
        return recommendations[:10]  # Limit to top 10
    
    def _enhanced_risk_assessment(self) -> Dict[str, Any]:
        """Enhanced risk assessment with LLM context."""
        base_risk = {
            "level": self.algorithmic_result.safety_level,
            "score": self.algorithmic_result.score,
            "warnings": self.algorithmic_result.warnings
        }
        
        if self.llm_insights.get("risk_factors"):
            base_risk["llm_risk_factors"] = self.llm_insights["risk_factors"]
            base_risk["contextual_risks"] = self.llm_insights.get("contextual_risks", [])
        
        return base_risk
    
    def _extract_personalized_insights(self) -> Dict[str, Any]:
        """Extract personalized insights from LLM analysis."""
        return {
            "timing_recommendations": self.llm_insights.get("optimal_timing", {}),
            "portion_guidance": self.llm_insights.get("portion_recommendations", {}),
            "preparation_tips": self.llm_insights.get("preparation_suggestions", []),
            "alternatives": self.llm_insights.get("healthier_alternatives", []),
            "interaction_warnings": self.llm_insights.get("interaction_warnings", [])
        }


class HybridHealthAnalyzer:
    """Hybrid analyzer combining clinical algorithms with LLM intelligence."""
    
    def __init__(self, base_analyzer: BaseHealthAnalyzer):
        self.base_analyzer = base_analyzer
        self.profile_type = base_analyzer.profile_type
    
    async def analyze_hybrid(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        severity: Severity,
        insight_level: Optional[ContextualInsightLevel] = None
    ) -> EnhancedAnalysisResult:
        """
        Perform intelligent hybrid analysis with context-aware AI enhancement.
        
        Args:
            nutrition_data: Product nutrition information
            user_context: User's health profile and demographics
            severity: Condition severity level
            insight_level: Optional override for AI enhancement level
            
        Returns:
            EnhancedAnalysisResult with optimal clinical and LLM insights
        """
        logger.info(
            "Starting intelligent hybrid analysis",
            profile=self.profile_type.value,
            insight_level=insight_level.value if insight_level else "auto-select",
            auto_selected=insight_level is None
        )
        # STEP 1: Clinical Algorithmic Analysis (Foundation)
        algorithmic_result = await self.base_analyzer.analyze(
            nutrition_data, user_context, severity
        )
        
        # STEP 2: LLM Enhancement (if enabled)
        llm_insights = {}
        
        # Auto-select insight level if not provided
        if insight_level is None:
            # Create temporary factory instance for insight level determination
            factory = EnhancedHealthAnalyzerFactory()
            insight_level = factory._determine_optimal_insight_level(
                nutrition_data, user_context, severity
            )
        hybrid_score = algorithmic_result.score
        
        if insight_level != ContextualInsightLevel.CLINICAL_FOCUS:
            llm_insights = await self._get_llm_enhancement(
                nutrition_data, user_context, algorithmic_result, insight_level
            )
            
            # STEP 3: Score Fusion
            hybrid_score = self._calculate_hybrid_score(
                algorithmic_result.score,
                llm_insights,
                insight_level
            )
        
        # STEP 4: Confidence Assessment
        confidence_metrics = self._calculate_confidence_metrics(
            algorithmic_result, llm_insights, insight_level
        )
        
        # Skip LLM enhancement for clinical-focus mode
        if insight_level == ContextualInsightLevel.CLINICAL_FOCUS:
            return EnhancedAnalysisResult(
                algorithmic_result,
                llm_insights={"insight_level_used": insight_level.value},
                hybrid_score=algorithmic_result.score,
                confidence_metrics=confidence_metrics
            )
        
        # Add insight level to LLM insights for tracking
        if llm_insights:
            llm_insights["insight_level_used"] = insight_level.value
        else:
            llm_insights = {"insight_level_used": insight_level.value}
        
        return EnhancedAnalysisResult(
            algorithmic_result,
            llm_insights=llm_insights,
            hybrid_score=hybrid_score,
            confidence_metrics=confidence_metrics
        )
    
    async def _get_llm_enhancement(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        algorithmic_result: AnalysisResult,
        insight_level: ContextualInsightLevel
    ) -> Dict[str, Any]:
        """Get LLM enhancement for the analysis."""
        
        # Build context-aware enhancement prompt
        prompt = self._build_contextual_prompt(
            nutrition_data, user_context, algorithmic_result, insight_level
        )      
        
        try:
            # Use Claude/GPT for advanced reasoning
            llm_response = await enhance_nutrition_analysis(
                nutrition_data=nutrition_data.__dict__,
                health_context=user_context.__dict__,
                analysis_prompt=prompt,
                insight_level=insight_level.value
            )
            
            # Parse and structure LLM insights
            return self._parse_llm_insights(llm_response, insight_level)
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return {"error": str(e), "fallback_used": True}
    
    def _build_contextual_prompt(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        algorithmic_result: AnalysisResult,
        insight_level: ContextualInsightLevel
    ) -> str:
        """Build context-aware enhancement prompt."""
        
        base_prompt = f"""
You are an expert clinical nutritionist analyzing food products for {self.profile_type.value} management.

PRODUCT ANALYSIS:
- Product: {nutrition_data.product_name}
- Algorithmic Score: {algorithmic_result.score}/100
- Safety Level: {algorithmic_result.safety_level}
- Key Warnings: {algorithmic_result.warnings}

NUTRITION DATA:
- Calories: {nutrition_data.calories}
- Carbs: {nutrition_data.carbohydrates}g (Sugar: {nutrition_data.sugar}g)
- Protein: {nutrition_data.protein}g
- Fat: {nutrition_data.total_fat}g
- Fiber: {nutrition_data.fiber}g
- Sodium: {nutrition_data.sodium}mg

USER CONTEXT:
- Health Profile: {self.profile_type.value}
- Severity: {user_context.primary_profiles[0].severity.value if user_context.primary_profiles else 'moderate'}
- Age Group: {user_context.age_group.value}
- Activity Level: {user_context.activity_level.value}
- Goals: {[p.goals for p in user_context.primary_profiles if p.profile_type == self.profile_type][0] if user_context.primary_profiles else []}
"""
        
        if insight_level == ContextualInsightLevel.SMART_ENHANCEMENT:
            base_prompt += """
TASK: Enhance the algorithmic analysis with contextual insights.
Focus on:
1. Portion size recommendations for this user
2. Optimal timing for consumption
3. Preparation methods to improve health impact
4. Specific warnings for this user's profile
5. Healthier alternatives

Adjust the algorithmic score by ±15 points maximum based on context.
"""
        
        elif insight_level == ContextualInsightLevel.BALANCED_INTELLIGENCE:
            base_prompt += """
TASK: Provide balanced hybrid analysis combining clinical algorithms with contextual intelligence.
Focus on:
1. Validate/challenge algorithmic scoring with clinical reasoning
2. Personalized recommendations based on user's specific context
3. Risk assessment considering user's complete health profile
4. Practical guidance for real-world consumption
5. Long-term health impact assessment

Adjust the algorithmic score by ±25 points based on comprehensive analysis.
"""
        
        elif insight_level == ContextualInsightLevel.MAXIMUM_INSIGHT:
            base_prompt += """
TASK: Provide comprehensive clinical analysis using the algorithmic result as reference only.
Focus on:
1. Complete re-evaluation using clinical expertise
2. Holistic health impact assessment
3. Personalized risk-benefit analysis
4. Comprehensive lifestyle integration recommendations
5. Evidence-based clinical reasoning

Provide independent scoring with full clinical justification.
"""
        
        
        base_prompt += """
RESPONSE FORMAT (JSON):
{
  "score_adjustment": -10,
  "adjustment_reasoning": "Portion control reduces glycemic impact",
  "risk_factors": ["High sugar content", "Low fiber"],
  "contextual_risks": ["May spike glucose if consumed alone"],
  "optimal_timing": {"best": "morning", "avoid": "evening"},
  "portion_recommendations": {"max_serving": "15g", "frequency": "occasional"},
  "preparation_suggestions": ["Combine with nuts", "Add fiber"],
  "healthier_alternatives": ["Almond butter", "Tahini"],
  "interaction_warnings": ["Monitor glucose closely"],
  "clinical_reasoning": "Evidence-based explanation"
}
"""
        
        return base_prompt
    
    def _parse_llm_insights(self, llm_response: Dict[str, Any], insight_level: ContextualInsightLevel) -> Dict[str, Any]:
        """Parse LLM response into structured insights."""
        ai_insights = llm_response.get("ai_insights", {})
        
        return {
            "enhanced_score": ai_insights.get("enhanced_score", 0),
            "reasoning": ai_insights.get("reasoning", ""),
            "recommendations": ai_insights.get("recommendations", []),
            "alternatives": ai_insights.get("alternatives", []),
            "timing_advice": ai_insights.get("timing_advice", ""),
            "risk_factors": ai_insights.get("risk_factors", []),
            "confidence": ai_insights.get("confidence", 0.7),
            "model_used": llm_response.get("model_used", "mock"),
            "usage": llm_response.get("usage", {})
        }
    
    def _calculate_hybrid_score(self, algorithmic_score: int, llm_insights: Dict[str, Any], insight_level: ContextualInsightLevel) -> int:
        """Calculate hybrid score combining algorithmic and LLM insights."""
        if not llm_insights or "error" in llm_insights:
            return algorithmic_score
        
        enhancement = llm_insights.get("enhanced_score", 0)
        
        # Apply enhancement based on insight level
        if insight_level == ContextualInsightLevel.SMART_ENHANCEMENT:
            adjustment = min(max(enhancement, -15), 15)  # ±15 points max
        elif insight_level == ContextualInsightLevel.BALANCED_INTELLIGENCE:
            adjustment = min(max(enhancement, -25), 25)  # ±25 points max
        elif insight_level == ContextualInsightLevel.MAXIMUM_INSIGHT:
            adjustment = min(max(enhancement, -50), 50)  # ±50 points max
        else:
            adjustment = 0
        
        hybrid_score = algorithmic_score + adjustment
        return min(max(hybrid_score, 0), 100)  # Keep within 0-100 range
    
    def _calculate_confidence_metrics(self, algorithmic_result: AnalysisResult, llm_insights: Dict[str, Any], insight_level: ContextualInsightLevel) -> Dict[str, float]:
        """Calculate confidence metrics for the hybrid analysis."""
        metrics = {
            "algorithmic_confidence": 0.85,
            "llm_confidence": llm_insights.get("confidence", 0.0) if llm_insights and "error" not in llm_insights else 0.0,
            "hybrid_confidence": 0.85,
            "data_quality": 0.8
        }
        
        # Calculate overall confidence based on insight level
        if llm_insights and "error" not in llm_insights:
            llm_conf = metrics["llm_confidence"]
            alg_conf = metrics["algorithmic_confidence"]
            
            if insight_level == ContextualInsightLevel.SMART_ENHANCEMENT:
                overall = (alg_conf * 0.8) + (llm_conf * 0.2)
            elif insight_level == ContextualInsightLevel.BALANCED_INTELLIGENCE:
                overall = (alg_conf * 0.6) + (llm_conf * 0.4)
            elif insight_level == ContextualInsightLevel.MAXIMUM_INSIGHT:
                overall = (alg_conf * 0.3) + (llm_conf * 0.7)
            else:
                overall = alg_conf
            
            metrics["hybrid_confidence"] = overall
        
        return metrics
    
    def _calculate_risk_score(self, nutrition_data: NutritionData, severity: Severity) -> int:
        """Calculate risk score based on nutrition data and user severity (0-100)."""
        risk = 0
        
        # High sugar content
        if nutrition_data.sugar > 15:
            risk += 30
        elif nutrition_data.sugar > 10:
            risk += 20
        elif nutrition_data.sugar > 5:
            risk += 10
        
        # High sodium
        if nutrition_data.sodium > 800:
            risk += 25
        elif nutrition_data.sodium > 400:
            risk += 15
        
        # Low fiber (processed indicator)
        if nutrition_data.fiber < 1:
            risk += 15
        elif nutrition_data.fiber < 3:
            risk += 10
        
        # High calories per serving
        if nutrition_data.calories > 300:
            risk += 15
        elif nutrition_data.calories > 200:
            risk += 10
        
        # Severity multiplier
        if severity == Severity.SEVERE:
            risk = int(risk * 1.5)
        elif severity == Severity.MODERATE:
            risk = int(risk * 1.2)
        
        return min(risk, 100)
    
    def _calculate_product_complexity(self, nutrition_data: NutritionData) -> int:
        """Calculate how complex/processed a product is (0-100)."""
        complexity = 0
        
        # High sugar content
        if nutrition_data.sugar > 10:
            complexity += 25
        elif nutrition_data.sugar > 5:
            complexity += 15
        
        # High sodium
        if nutrition_data.sodium > 600:
            complexity += 20
        elif nutrition_data.sodium > 300:
            complexity += 10
        
        # Low fiber (processed indicator)
        if nutrition_data.fiber < 2:
            complexity += 15
        
        # High saturated fat
        if hasattr(nutrition_data, 'saturated_fat') and nutrition_data.saturated_fat > 5:
            complexity += 15
        
        # Many ingredients (if available)
        if hasattr(nutrition_data, 'ingredients') and len(nutrition_data.ingredients) > 10:
            complexity += 15
        
        return min(complexity, 100)
    
    def _calculate_user_needs(self, user_context: UserHealthContext) -> int:
        """Calculate how much guidance user needs (0-100)."""
        needs_score = 0
        
        # Multiple conditions need more guidance
        if len(user_context.primary_profiles) > 1:
            needs_score += 25
        
        # Severe conditions need more guidance
        for profile in user_context.primary_profiles:
            if profile.severity == Severity.SEVERE:
                needs_score += 30
            elif profile.severity == Severity.MODERATE:
                needs_score += 20
            else:
                needs_score += 10
        
        # Age factors (older users may need more guidance)
        if hasattr(user_context, 'age_group'):
            if user_context.age_group.value in ['senior', 'elderly']:
                needs_score += 15
        
        return min(needs_score, 100)
    
    def _has_severe_conditions(self, user_context: UserHealthContext) -> bool:
        """Check if user has any severe health conditions."""
        return any(
            profile.severity == Severity.SEVERE 
            for profile in user_context.primary_profiles
        )
    
    def _calculate_confidence_metrics(
        self,
        algorithmic_result: AnalysisResult,
        llm_insights: Dict[str, Any],
        insight_level: ContextualInsightLevel
    ) -> Dict[str, float]:
        """Calculate confidence metrics for the hybrid analysis."""
        
        metrics = {
            "algorithmic_confidence": 0.85,  # Base algorithmic confidence
            "llm_confidence": 0.0,
            "hybrid_confidence": 0.85,
            "data_quality": 0.8
        }
        
        if llm_insights and "error" not in llm_insights:
            # LLM confidence based on response quality
            has_reasoning = bool(llm_insights.get("clinical_reasoning"))
            has_alternatives = bool(llm_insights.get("healthier_alternatives"))
            has_timing = bool(llm_insights.get("optimal_timing"))
            
            llm_confidence = (
                0.4 + 
                (0.3 if has_reasoning else 0) +
                (0.2 if has_alternatives else 0) +
                (0.1 if has_timing else 0)
            )
            
            metrics["llm_confidence"] = llm_confidence
            
            # Weight confidences based on insight level
            if insight_level == ContextualInsightLevel.SMART_ENHANCEMENT:
                overall = (metrics["algorithmic_confidence"] * 0.8) + (llm_confidence * 0.2)
            elif insight_level == ContextualInsightLevel.BALANCED_INTELLIGENCE:
                overall = (metrics["algorithmic_confidence"] * 0.6) + (llm_confidence * 0.4)
            elif insight_level == ContextualInsightLevel.MAXIMUM_INSIGHT:
                overall = (metrics["algorithmic_confidence"] * 0.3) + (llm_confidence * 0.7)
            else:
                overall = metrics["algorithmic_confidence"] * 0.8 + llm_confidence * 0.2
            
            metrics["hybrid_confidence"] = overall
        
        return metrics
    

class EnhancedHealthAnalyzerFactory:
    """Enhanced factory for creating hybrid health analyzers."""
    
    _analyzers = {
        ProfileType.DIABETES: DiabetesAnalyzer,
        ProfileType.HYPERTENSION: HypertensionAnalyzer,
        # Add other analyzers as they're implemented
    }
    
    @classmethod
    def get_hybrid_analyzer(cls, profile_type: ProfileType) -> HybridHealthAnalyzer:
        """Get hybrid analyzer for the specified health profile."""
        
        if profile_type not in cls._analyzers:
            raise ValueError(f"No analyzer available for profile type: {profile_type}")
        
        base_analyzer = cls._analyzers[profile_type]()
        return HybridHealthAnalyzer(base_analyzer)
    
    @classmethod
    async def analyze_for_user_enhanced(
        cls,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        insight_level: Optional[ContextualInsightLevel] = None
    ) -> Dict[ProfileType, EnhancedAnalysisResult]:
        """
        Perform enhanced analysis for all user's health profiles.
        
        Args:
            nutrition_data: Product nutrition information
            user_context: User's complete health context
            insight_level: Optional insight level (auto-selected if None)
            
        Returns:
            Dictionary mapping profile types to enhanced analysis results
        """
        results = {}
        
        # Get user's health profiles
        profiles_to_analyze = []
        for profile in user_context.primary_profiles:
            profiles_to_analyze.append(profile.profile_type)
        
        # Analyze for each profile
        for profile_type in profiles_to_analyze:
            try:
                hybrid_analyzer = cls.get_hybrid_analyzer(profile_type)
                
                # Find the severity for this specific profile
                profile_severity = Severity.MODERATE  # Default
                for profile in user_context.primary_profiles:
                    if profile.profile_type == profile_type:
                        profile_severity = profile.severity
                        break
                
                result = await hybrid_analyzer.analyze_hybrid(
                    nutrition_data=nutrition_data,
                    user_context=user_context,
                    severity=profile_severity,
                    insight_level=insight_level
                )
                
                results[profile_type] = result
                
                logger.info(f"Enhanced analysis completed for {profile_type.value}")
                
            except Exception as e:
                logger.error(f"Enhanced analysis failed for {profile_type.value}: {e}")
                # Fallback to basic analysis
                basic_analyzer = cls._analyzers[profile_type]()
                # Find severity for fallback
                fallback_severity = Severity.MODERATE
                for profile in user_context.primary_profiles:
                    if profile.profile_type == profile_type:
                        fallback_severity = profile.severity
                        break
                
                basic_result = await basic_analyzer.analyze(
                    nutrition_data, user_context, fallback_severity
                )
                results[profile_type] = EnhancedAnalysisResult(basic_result)
        
        return results
    
    def _determine_optimal_insight_level(
        self,
        nutrition_data: NutritionData,
        user_context: UserHealthContext,
        severity: Severity
    ) -> ContextualInsightLevel:
        """Automatically determine optimal AI insight level based on context."""
        
        # Calculate risk factors
        risk_score = self._calculate_risk_score(nutrition_data, severity)
        complexity_score = self._calculate_product_complexity(nutrition_data)
        user_needs_score = self._calculate_user_needs(user_context)
        
        logger.info(
            "Context assessment for auto-selection",
            risk_score=risk_score,
            complexity_score=complexity_score,
            user_needs_score=user_needs_score
        )
        
        # High-risk situations need maximum AI insight
        if (risk_score > 60 or severity == Severity.SEVERE or 
            self._has_severe_conditions(user_context)):
            return ContextualInsightLevel.MAXIMUM_INSIGHT
        
        # Complex products or high user needs = balanced intelligence
        elif complexity_score > 50 or user_needs_score > 60:
            return ContextualInsightLevel.BALANCED_INTELLIGENCE
        
        # Moderate risk = smart enhancement
        elif risk_score > 30 or complexity_score > 25:
            return ContextualInsightLevel.SMART_ENHANCEMENT
        
        # Low risk, simple products = clinical focus
        else:
            return ContextualInsightLevel.CLINICAL_FOCUS
    
    def _calculate_risk_score(self, nutrition_data: NutritionData, severity: Severity) -> int:
        """Calculate risk score based on nutrition data and user severity (0-100)."""
        risk = 0
        
        # High sugar content
        if nutrition_data.sugar and nutrition_data.sugar > 15:
            risk += 30
        elif nutrition_data.sugar and nutrition_data.sugar > 10:
            risk += 20
        elif nutrition_data.sugar and nutrition_data.sugar > 5:
            risk += 10
        
        # High sodium content
        if nutrition_data.sodium and nutrition_data.sodium > 800:
            risk += 25
        elif nutrition_data.sodium and nutrition_data.sodium > 400:
            risk += 15
        elif nutrition_data.sodium and nutrition_data.sodium > 200:
            risk += 10
        
        # Low fiber content
        if nutrition_data.fiber and nutrition_data.fiber < 1:
            risk += 15
        elif nutrition_data.fiber and nutrition_data.fiber < 3:
            risk += 10
        
        # High calories per serving
        if nutrition_data.calories > 300:
            risk += 15
        elif nutrition_data.calories > 200:
            risk += 10
        
        # Severity multiplier
        if severity == Severity.SEVERE:
            risk = int(risk * 1.5)
        elif severity == Severity.MODERATE:
            risk = int(risk * 1.2)
        
        return min(risk, 100)
    
    def _calculate_product_complexity(self, nutrition_data: NutritionData) -> int:
        """Calculate how complex/processed a product is (0-100)."""
        complexity = 0
        
        # High sugar content
        if nutrition_data.sugar and nutrition_data.sugar > 10:
            complexity += 20
        
        # High sodium content
        if nutrition_data.sodium and nutrition_data.sodium > 400:
            complexity += 20
        
        # Low fiber (processed foods)
        if nutrition_data.fiber and nutrition_data.fiber < 2:
            complexity += 15
        
        # High calorie density
        if nutrition_data.calories > 250:
            complexity += 15
        
        # Trans fat presence
        if nutrition_data.trans_fat and nutrition_data.trans_fat > 0:
            complexity += 30
        
        return min(complexity, 100)
    
    def _calculate_user_needs(self, user_context: UserHealthContext) -> int:
        """Calculate how much guidance user needs (0-100)."""
        needs_score = 0
        
        # Multiple conditions need more guidance
        if len(user_context.primary_profiles) > 1:
            needs_score += 25
        
        # Severe conditions need more guidance
        for profile in user_context.primary_profiles:
            if profile.severity == Severity.SEVERE:
                needs_score += 30
            elif profile.severity == Severity.MODERATE:
                needs_score += 20
            else:
                needs_score += 10
        
        # Age factors (older users may need more guidance)
        if hasattr(user_context, 'age_group'):
            if user_context.age_group.value in ['senior', 'elderly']:
                needs_score += 15
        
        return min(needs_score, 100)
    
    def _has_severe_conditions(self, user_context: UserHealthContext) -> bool:
        """Check if user has any severe health conditions."""
        return any(
            profile.severity == Severity.SEVERE 
            for profile in user_context.primary_profiles
        )
    
    @classmethod
    def get_overall_enhanced_score(
        cls, 
        analysis_results: Dict[ProfileType, EnhancedAnalysisResult]
    ) -> int:
        """Calculate overall enhanced score from multiple profile analyses."""
        
        if not analysis_results:
            return 50  # Default neutral score
        
        # Weight primary condition more heavily
        scores = []
        weights = []
        
        for profile_type, result in analysis_results.items():
            scores.append(result.final_score)
            # Primary condition gets higher weight
            weight = 0.7 if len(analysis_results) == 1 else 0.5
            weights.append(weight)
        
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return int(weighted_sum / total_weight) if total_weight > 0 else 50


# Global enhanced factory instance
enhanced_analyzer_factory = EnhancedHealthAnalyzerFactory()
