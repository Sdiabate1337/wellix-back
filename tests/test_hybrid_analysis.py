"""
Comprehensive test suite for hybrid LLM analysis with real nutrition data.
Tests all integration levels and validates LLM insights quality.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import structlog

# Import Wellix components
from app.models.health import (
    NutritionData, UserHealthContext, HealthProfile, 
    ProfileType, Severity, AgeGroup, ActivityLevel, Demographics
)
from app.services.health_analyzers.analyzer_factory import (
    EnhancedHealthAnalyzerFactory, LLMIntegrationLevel
)
from app.workflows.food_analysis_workflow import food_analysis_workflow

logger = structlog.get_logger(__name__)


class HybridAnalysisTestSuite:
    """Comprehensive test suite for hybrid LLM analysis."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
    async def run_all_tests(self):
        """Run complete test suite for hybrid analysis."""
        print("🚀 Starting Hybrid Analysis Test Suite")
        print("=" * 60)
        
        # Test 1: Sample nutrition data preparation
        print("\n📋 Step 1: Preparing sample nutrition data...")
        sample_products = self._prepare_sample_products()
        print(f"✅ Prepared {len(sample_products)} sample products")
        
        # Test 2: User health contexts
        print("\n👤 Step 2: Creating test user profiles...")
        test_users = self._create_test_users()
        print(f"✅ Created {len(test_users)} test user profiles")
        
        # Test 3: Test all integration levels
        print("\n🧠 Step 3: Testing all LLM integration levels...")
        await self._test_all_integration_levels(sample_products[0], test_users[0])
        
        # Test 4: Product variety testing
        print("\n🥗 Step 4: Testing product variety...")
        await self._test_product_variety(sample_products, test_users[0])
        
        # Test 5: User profile variety testing
        print("\n👥 Step 5: Testing user profile variety...")
        await self._test_user_variety(sample_products[0], test_users)
        
        # Test 6: Performance benchmarking
        print("\n⚡ Step 6: Performance benchmarking...")
        await self._benchmark_performance(sample_products[0], test_users[0])
        
        # Test 7: LLM insights quality validation
        print("\n🔍 Step 7: Validating LLM insights quality...")
        await self._validate_llm_quality(sample_products, test_users[0])
        
        # Generate final report
        print("\n📊 Generating test report...")
        self._generate_test_report()
        
        print("\n🎉 Hybrid Analysis Test Suite Completed!")
        return self.test_results
    
    def _prepare_sample_products(self) -> List[NutritionData]:
        """Prepare realistic nutrition data samples."""
        
        # Sample 1: High-sodium processed snack (challenging for hypertension)
        nutella = NutritionData(
            product_name="Nutella Hazelnut Spread",
            brand="Ferrero",
            barcode="8000500037003",
            serving_size="15g (1 tbsp)",
            calories=80,
            protein=1.0,
            carbohydrates=8.5,
            sugar=8.0,
            total_fat=4.5,
            saturated_fat=1.5,
            fiber=0.0,
            sodium=5,  # Low sodium but high sugar
            potassium=58,
            cholesterol=0,
            ingredients=[
                "Sugar", "Palm Oil", "Hazelnuts", "Cocoa", "Skim Milk Powder",
                "Reduced Minerals Whey Powder", "Lecithin", "Vanillin"
            ],
            allergens=["Milk", "Tree Nuts (Hazelnuts)", "May contain Gluten"],
            additives=["Lecithin (E322)", "Vanillin"],
            nutrition_grade="D",
            nova_group=4
        )
        
        # Sample 2: Healthy but high-sodium option (complex for analysis)
        canned_salmon = NutritionData(
            product_name="Wild Alaskan Salmon",
            brand="Bumble Bee",
            barcode="8660000020",
            serving_size="85g (1/2 cup)",
            calories=110,
            protein=20.0,
            carbohydrates=0.0,
            sugar=0.0,
            total_fat=3.0,
            saturated_fat=1.0,
            fiber=0.0,
            sodium=380,  # High sodium but healthy protein
            potassium=350,
            cholesterol=40,
            ingredients=["Wild Alaskan Salmon", "Salt"],
            allergens=["Fish"],
            additives=[],
            nutrition_grade="B",
            nova_group=1
        )
        
        # Sample 3: Complex processed food (multiple health considerations)
        frozen_pizza = NutritionData(
            product_name="Margherita Pizza",
            brand="DiGiorno",
            barcode="7123456789",
            serving_size="1/4 pizza (120g)",
            calories=290,
            protein=12.0,
            carbohydrates=35.0,
            sugar=4.0,
            total_fat=11.0,
            saturated_fat=5.0,
            fiber=2.0,
            sodium=680,  # High sodium
            potassium=200,
            cholesterol=20,
            ingredients=[
                "Enriched Wheat Flour", "Water", "Mozzarella Cheese", "Tomato Paste",
                "Soybean Oil", "Salt", "Yeast", "Sugar", "Garlic Powder", "Oregano",
                "Calcium Propionate", "Sodium Benzoate"
            ],
            allergens=["Wheat", "Milk", "Soy"],
            additives=["Calcium Propionate (E282)", "Sodium Benzoate (E211)"],
            nutrition_grade="C",
            nova_group=4
        )
        
        return [nutella, canned_salmon, frozen_pizza]
    
    def _create_test_users(self) -> List[UserHealthContext]:
        """Create diverse test user profiles."""
        
        # User 1: Diabetes Type 2, Moderate severity
        diabetes_user = UserHealthContext(
            user_id="test_user_diabetes",
            health_profile=HealthProfile(
                primary_condition=ProfileType.DIABETES,
                severity=Severity.MODERATE,
                secondary_conditions=[],
                restrictions=["Low sugar", "Controlled carbs"],
                goals=["Maintain HbA1c < 7%", "Weight management"],
                medications=["Metformin", "Insulin"],
                target_values={"hba1c": 6.5, "glucose_fasting": 100}
            ),
            demographics=Demographics(
                age_group=AgeGroup.MIDDLE_AGED,
                activity_level=ActivityLevel.LIGHTLY_ACTIVE,
                weight_kg=78.0,
                height_cm=170.0
            ),
            allergies=["Shellfish"],
            dietary_preferences=["Low carb"],
            weight_goals="maintain",
            preferred_language="en",
            analysis_depth="detailed"
        )
        
        # User 2: Hypertension, High severity
        hypertension_user = UserHealthContext(
            user_id="test_user_hypertension",
            health_profile=HealthProfile(
                primary_condition=ProfileType.HYPERTENSION,
                severity=Severity.HIGH,
                secondary_conditions=[],
                restrictions=["Low sodium", "DASH diet"],
                goals=["BP < 130/80", "Reduce medication dependency"],
                medications=["Lisinopril", "Hydrochlorothiazide"],
                target_values={"systolic_bp": 125, "diastolic_bp": 75}
            ),
            demographics=Demographics(
                age_group=AgeGroup.SENIOR,
                activity_level=ActivityLevel.MODERATELY_ACTIVE,
                weight_kg=85.0,
                height_cm=175.0
            ),
            allergies=[],
            dietary_preferences=["Heart healthy", "Low sodium"],
            weight_goals="lose",
            preferred_language="en",
            analysis_depth="comprehensive"
        )
        
        # User 3: Multiple conditions (Diabetes + Hypertension)
        combined_user = UserHealthContext(
            user_id="test_user_combined",
            health_profile=HealthProfile(
                primary_condition=ProfileType.DIABETES,
                severity=Severity.HIGH,
                secondary_conditions=[ProfileType.HYPERTENSION],
                restrictions=["Low sodium", "Low sugar", "Controlled portions"],
                goals=["HbA1c < 6.5%", "BP < 130/80", "Weight loss"],
                medications=["Metformin", "Insulin", "Lisinopril"],
                target_values={"hba1c": 6.5, "systolic_bp": 125}
            ),
            demographics=Demographics(
                age_group=AgeGroup.MIDDLE_AGED,
                activity_level=ActivityLevel.SEDENTARY,
                weight_kg=95.0,
                height_cm=168.0
            ),
            allergies=["Nuts"],
            dietary_preferences=["Diabetic friendly", "Heart healthy"],
            weight_goals="lose",
            preferred_language="en",
            analysis_depth="comprehensive"
        )
        
        return [diabetes_user, hypertension_user, combined_user]
    
    async def _test_all_integration_levels(self, product: NutritionData, user: UserHealthContext):
        """Test all 4 LLM integration levels with the same product and user."""
        
        print(f"  Testing product: {product.product_name}")
        print(f"  User profile: {user.health_profile.primary_condition.value}")
        
        for level in LLMIntegrationLevel:
            print(f"\n  🧠 Testing {level.value}...")
            
            start_time = time.time()
            
            try:
                # Test direct analyzer factory
                results = await EnhancedHealthAnalyzerFactory.analyze_for_user_enhanced(
                    nutrition_data=product,
                    user_context=user,
                    integration_level=level
                )
                
                processing_time = time.time() - start_time
                
                # Analyze results
                test_result = {
                    "test_type": "integration_level",
                    "integration_level": level.value,
                    "product": product.product_name,
                    "user_profile": user.health_profile.primary_condition.value,
                    "processing_time_seconds": round(processing_time, 2),
                    "success": True,
                    "profiles_analyzed": len(results),
                    "enhanced_analysis": level != LLMIntegrationLevel.ALGORITHMIC_ONLY,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Extract key metrics
                for profile_type, result in results.items():
                    profile_key = f"{profile_type.value}_analysis"
                    test_result[profile_key] = {
                        "algorithmic_score": result.algorithmic_result.score,
                        "hybrid_score": result.final_score,
                        "score_adjustment": result.final_score - result.algorithmic_result.score,
                        "has_llm_insights": bool(result.llm_insights and "error" not in result.llm_insights),
                        "confidence_metrics": result.confidence_metrics,
                        "enhanced_recommendations_count": len(result.enhanced_recommendations)
                    }
                    
                    if result.llm_insights and "error" not in result.llm_insights:
                        test_result[profile_key]["llm_insights"] = {
                            "risk_factors_count": len(result.llm_insights.get("risk_factors", [])),
                            "alternatives_count": len(result.llm_insights.get("healthier_alternatives", [])),
                            "has_timing_recommendations": bool(result.llm_insights.get("optimal_timing")),
                            "clinical_reasoning_length": len(result.llm_insights.get("clinical_reasoning", ""))
                        }
                
                self.test_results.append(test_result)
                
                print(f"    ✅ Success - {processing_time:.2f}s")
                print(f"    📊 Profiles: {len(results)}, Enhanced: {test_result['enhanced_analysis']}")
                
            except Exception as e:
                error_result = {
                    "test_type": "integration_level",
                    "integration_level": level.value,
                    "product": product.product_name,
                    "user_profile": user.health_profile.primary_condition.value,
                    "processing_time_seconds": time.time() - start_time,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.test_results.append(error_result)
                print(f"    ❌ Error: {str(e)}")
    
    async def _test_product_variety(self, products: List[NutritionData], user: UserHealthContext):
        """Test different product types with hybrid_balanced level."""
        
        print(f"  Testing with user: {user.health_profile.primary_condition.value}")
        
        for product in products:
            print(f"\n  🥗 Testing {product.product_name}...")
            
            start_time = time.time()
            
            try:
                results = await EnhancedHealthAnalyzerFactory.analyze_for_user_enhanced(
                    nutrition_data=product,
                    user_context=user,
                    integration_level=LLMIntegrationLevel.HYBRID_BALANCED
                )
                
                processing_time = time.time() - start_time
                
                # Calculate overall impact
                overall_score = EnhancedHealthAnalyzerFactory.get_overall_enhanced_score(results)
                
                test_result = {
                    "test_type": "product_variety",
                    "product": product.product_name,
                    "product_category": self._categorize_product(product),
                    "user_profile": user.health_profile.primary_condition.value,
                    "processing_time_seconds": round(processing_time, 2),
                    "overall_score": overall_score,
                    "nutrition_complexity": self._assess_nutrition_complexity(product),
                    "llm_insights_quality": self._assess_llm_insights_quality(results),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.test_results.append(test_result)
                
                print(f"    ✅ Score: {overall_score}/100, Time: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
    
    async def _test_user_variety(self, product: NutritionData, users: List[UserHealthContext]):
        """Test same product with different user profiles."""
        
        print(f"  Testing product: {product.product_name}")
        
        for user in users:
            print(f"\n  👤 Testing user: {user.health_profile.primary_condition.value}")
            
            start_time = time.time()
            
            try:
                results = await EnhancedHealthAnalyzerFactory.analyze_for_user_enhanced(
                    nutrition_data=product,
                    user_context=user,
                    integration_level=LLMIntegrationLevel.HYBRID_BALANCED
                )
                
                processing_time = time.time() - start_time
                overall_score = EnhancedHealthAnalyzerFactory.get_overall_enhanced_score(results)
                
                test_result = {
                    "test_type": "user_variety",
                    "product": product.product_name,
                    "user_profile": user.health_profile.primary_condition.value,
                    "user_severity": user.health_profile.severity.value,
                    "secondary_conditions": [c.value for c in user.health_profile.secondary_conditions],
                    "processing_time_seconds": round(processing_time, 2),
                    "overall_score": overall_score,
                    "personalization_level": self._assess_personalization_level(results, user),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.test_results.append(test_result)
                
                print(f"    ✅ Score: {overall_score}/100, Personalization: {test_result['personalization_level']}")
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
    
    async def _benchmark_performance(self, product: NutritionData, user: UserHealthContext):
        """Benchmark performance across integration levels."""
        
        print("  Running performance benchmarks...")
        
        benchmark_results = {}
        
        for level in LLMIntegrationLevel:
            times = []
            
            # Run multiple iterations for accurate timing
            for i in range(3):
                start_time = time.time()
                
                try:
                    await EnhancedHealthAnalyzerFactory.analyze_for_user_enhanced(
                        nutrition_data=product,
                        user_context=user,
                        integration_level=level
                    )
                    
                    times.append(time.time() - start_time)
                    
                except Exception as e:
                    print(f"    ❌ Benchmark error for {level.value}: {e}")
                    continue
            
            if times:
                benchmark_results[level.value] = {
                    "avg_time_seconds": round(sum(times) / len(times), 2),
                    "min_time_seconds": round(min(times), 2),
                    "max_time_seconds": round(max(times), 2),
                    "iterations": len(times)
                }
                
                print(f"    ⚡ {level.value}: {benchmark_results[level.value]['avg_time_seconds']}s avg")
        
        self.performance_metrics = benchmark_results
    
    async def _validate_llm_quality(self, products: List[NutritionData], user: UserHealthContext):
        """Validate quality of LLM insights."""
        
        print("  Validating LLM insights quality...")
        
        quality_metrics = {
            "total_analyses": 0,
            "successful_llm_insights": 0,
            "insights_with_reasoning": 0,
            "insights_with_alternatives": 0,
            "insights_with_timing": 0,
            "average_insight_length": 0
        }
        
        total_insight_length = 0
        
        for product in products:
            try:
                results = await EnhancedHealthAnalyzerFactory.analyze_for_user_enhanced(
                    nutrition_data=product,
                    user_context=user,
                    integration_level=LLMIntegrationLevel.LLM_DOMINANT
                )
                
                quality_metrics["total_analyses"] += 1
                
                for profile_type, result in results.items():
                    if result.llm_insights and "error" not in result.llm_insights:
                        quality_metrics["successful_llm_insights"] += 1
                        
                        insights = result.llm_insights
                        
                        if insights.get("clinical_reasoning"):
                            quality_metrics["insights_with_reasoning"] += 1
                            total_insight_length += len(insights["clinical_reasoning"])
                        
                        if insights.get("healthier_alternatives"):
                            quality_metrics["insights_with_alternatives"] += 1
                        
                        if insights.get("optimal_timing"):
                            quality_metrics["insights_with_timing"] += 1
                
            except Exception as e:
                print(f"    ❌ Quality validation error: {e}")
        
        if quality_metrics["insights_with_reasoning"] > 0:
            quality_metrics["average_insight_length"] = round(
                total_insight_length / quality_metrics["insights_with_reasoning"]
            )
        
        self.test_results.append({
            "test_type": "llm_quality_validation",
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"    📊 Success rate: {quality_metrics['successful_llm_insights']}/{quality_metrics['total_analyses']}")
    
    def _categorize_product(self, product: NutritionData) -> str:
        """Categorize product type for analysis."""
        if product.nova_group >= 4:
            return "ultra_processed"
        elif product.nova_group == 3:
            return "processed"
        elif product.sugar > 10:
            return "high_sugar"
        elif product.sodium > 400:
            return "high_sodium"
        elif product.protein > 15:
            return "high_protein"
        else:
            return "standard"
    
    def _assess_nutrition_complexity(self, product: NutritionData) -> str:
        """Assess nutritional complexity of product."""
        complexity_score = 0
        
        if product.sodium > 300: complexity_score += 1
        if product.sugar > 5: complexity_score += 1
        if len(product.additives) > 2: complexity_score += 1
        if len(product.allergens) > 1: complexity_score += 1
        if product.nova_group >= 3: complexity_score += 1
        
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_llm_insights_quality(self, results: Dict[Any, Any]) -> str:
        """Assess quality of LLM insights."""
        quality_score = 0
        total_profiles = len(results)
        
        for result in results.values():
            if result.llm_insights and "error" not in result.llm_insights:
                insights = result.llm_insights
                
                if insights.get("clinical_reasoning"): quality_score += 1
                if insights.get("healthier_alternatives"): quality_score += 1
                if insights.get("optimal_timing"): quality_score += 1
                if len(insights.get("risk_factors", [])) > 0: quality_score += 1
        
        avg_quality = quality_score / (total_profiles * 4) if total_profiles > 0 else 0
        
        if avg_quality >= 0.8:
            return "excellent"
        elif avg_quality >= 0.6:
            return "good"
        elif avg_quality >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _assess_personalization_level(self, results: Dict[Any, Any], user: UserHealthContext) -> str:
        """Assess level of personalization in results."""
        personalization_score = 0
        
        for result in results.values():
            insights = result.personalized_insights
            
            if insights.get("timing_recommendations"): personalization_score += 1
            if insights.get("portion_guidance"): personalization_score += 1
            if insights.get("preparation_tips"): personalization_score += 1
            if insights.get("alternatives"): personalization_score += 1
            if insights.get("interaction_warnings"): personalization_score += 1
        
        if personalization_score >= 4:
            return "high"
        elif personalization_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        
        print("\n" + "=" * 60)
        print("📊 HYBRID ANALYSIS TEST REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get("success", True)])
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total tests run: {total_tests}")
        print(f"  Successful tests: {successful_tests}")
        print(f"  Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Performance metrics
        if self.performance_metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            for level, metrics in self.performance_metrics.items():
                print(f"  {level}: {metrics['avg_time_seconds']}s avg ({metrics['min_time_seconds']}-{metrics['max_time_seconds']}s)")
        
        # Integration level analysis
        level_tests = [r for r in self.test_results if r.get("test_type") == "integration_level"]
        if level_tests:
            print(f"\n🧠 INTEGRATION LEVEL ANALYSIS:")
            for level in LLMIntegrationLevel:
                level_results = [r for r in level_tests if r.get("integration_level") == level.value]
                if level_results:
                    avg_time = sum(r["processing_time_seconds"] for r in level_results) / len(level_results)
                    print(f"  {level.value}: {len(level_results)} tests, {avg_time:.2f}s avg")
        
        # Quality metrics
        quality_tests = [r for r in self.test_results if r.get("test_type") == "llm_quality_validation"]
        if quality_tests:
            quality_data = quality_tests[0]["quality_metrics"]
            print(f"\n🔍 LLM QUALITY METRICS:")
            print(f"  Success rate: {quality_data['successful_llm_insights']}/{quality_data['total_analyses']}")
            print(f"  With reasoning: {quality_data['insights_with_reasoning']}")
            print(f"  With alternatives: {quality_data['insights_with_alternatives']}")
            print(f"  With timing: {quality_data['insights_with_timing']}")
        
        # Save detailed results
        with open("/Users/macook/Documents/wellix-back/test_results_hybrid_analysis.json", "w") as f:
            json.dump({
                "test_results": self.test_results,
                "performance_metrics": self.performance_metrics,
                "generated_at": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: test_results_hybrid_analysis.json")


async def main():
    """Run the hybrid analysis test suite."""
    test_suite = HybridAnalysisTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
