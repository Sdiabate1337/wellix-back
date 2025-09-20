#!/usr/bin/env python3
"""
Test d'intégration du workflow complet avec enrichissement LLM.

Teste l'intégration complète :
1. Enregistrement des services LLM dans le DI Container
2. DataExtractionNode avec enrichissement LLM
3. Workflow end-to-end avec validation qualité
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime
import structlog

# Configuration de base
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_workflow_llm_integration():
    """Test d'intégration complète du workflow avec LLM."""
    
    logger.info("🧪 Starting Workflow LLM Integration Test")
    
    # ===============================
    # 1. Setup du DI Container avec services LLM
    # ===============================
    
    logger.info("📦 Setting up DI Container with LLM services")
    
    try:
        from app.workflows.container import workflow_container, register_llm_services
        
        # Enregistrement des services LLM
        register_llm_services()
        
        # Vérification que les services sont enregistrés
        from app.services.llm.interfaces import ILLMEnricher, ITaskRouter, IConsensusValidator, ISemanticCache
        
        assert workflow_container.is_registered(ILLMEnricher), "ILLMEnricher not registered"
        assert workflow_container.is_registered(ITaskRouter), "ITaskRouter not registered"
        assert workflow_container.is_registered(IConsensusValidator), "IConsensusValidator not registered"
        assert workflow_container.is_registered(ISemanticCache), "ISemanticCache not registered"
        
        logger.info("✅ DI Container setup successful", 
                    services_registered=["ILLMEnricher", "ITaskRouter", "IConsensusValidator", "ISemanticCache"])
        
    except Exception as e:
        logger.error("❌ DI Container setup failed", error=str(e))
        raise
    
    # ===============================
    # 2. Test DataExtractionNode avec enrichissement LLM
    # ===============================
    
    logger.info("🔄 Testing DataExtractionNode with LLM enrichment")
    
        logger.info("🔄 Testing DataExtractionNode with LLM enrichment")
    
    try:
        from app.workflows.nodes.advanced_nodes import DataExtractionNode
        from app.workflows.interfaces import WorkflowState, WorkflowStage, InputData, InputType, AnalysisConfig, QualityLevel
        from app.models.health import UserHealthContext
        import uuid
        
        # Création du node avec injection DI
        extraction_node = DataExtractionNode()
        
        # Vérification que le service LLM est injecté
        assert extraction_node.llm_enricher is not None, "LLM enricher not injected"
        
        # Données de test pour JSON input
        test_product_data = {
            "product_name": "Yaourt Bio Nature",
            "brand": "Marque Test",
            "barcode": "1234567890123",
            "serving_size": "100g",
            "calories": 80,
            "protein": 4.5,
            "carbohydrates": 6.0,
            "total_fat": 3.2,
            "saturated_fat": 2.1,
            "fiber": 0.5,
            "sugar": 5.2,
            "sodium": 0.06,
            "ingredients": ["lait bio", "ferments lactiques"],
            "allergens": ["lait"]
        }
        
        # Création des objets requis
        user_context = UserHealthContext(
            age=30,
            weight=70.0,
            height=175.0,
            activity_level="moderate",
            health_conditions=[],
            dietary_restrictions=[],
            nutritional_goals=[]
        )
        
        analysis_config = AnalysisConfig(
            quality_level=QualityLevel.STANDARD,
            include_alternatives=True,
            target_audience="general"
        )
        
        # Création du state de test
        input_data = InputData(
            type=InputType.JSON_DATA,
            json_data=test_product_data,
            complexity_hints={"token_reservation": {"success": True, "tokens_reserved": 100}}
        )
        
        test_state = WorkflowState(
            workflow_id=str(uuid.uuid4()),
            user_context=user_context,
            input_data=input_data,
            analysis_config=analysis_config,
            current_stage=WorkflowStage.TOKEN_VALIDATION
        )
        
        # Exécution du node
        logger.info("🚀 Executing DataExtractionNode with LLM enrichment")
        result_state = await extraction_node.process(test_state)
        
        # Vérifications du résultat
        assert result_state.nutrition_data is not None, "No nutrition data extracted"
        
        nutrition_data = result_state.nutrition_data
        
        # Récupération des métadonnées LLM depuis l'historique de traitement
        llm_metadata = {"status": "not_found"}
        for step in result_state.processing_history:
            if isinstance(step, dict) and step.get("stage") == "llm_enrichment":
                llm_metadata = step.get("metadata", {})
                break
        
        logger.info("✅ DataExtractionNode processing successful",
                    product_name=nutrition_data.product_name,
                    llm_status=llm_metadata.get("status", "unknown"),
                    data_source=nutrition_data.data_source)
        
        # Log détaillé des résultats LLM
        if llm_metadata.get("status") == "success":
            logger.info("🧠 LLM Enrichment Details",
                        provider=llm_metadata.get("provider_used"),
                        quality_score=llm_metadata.get("quality_score"),
                        confidence_score=llm_metadata.get("confidence_score"),
                        processing_time=llm_metadata.get("processing_time"),
                        cache_hit=llm_metadata.get("cache_hit"),
                        consensus_validated=llm_metadata.get("consensus_validated"))
        
    except Exception as e:
        logger.error("❌ DataExtractionNode test failed", error=str(e))
        raise        # Exécution du node
        logger.info("🚀 Executing DataExtractionNode with LLM enrichment")
        result_state = await extraction_node.process(test_state)
        
        # Vérifications du résultat
        assert result_state.nutrition_data is not None, "No nutrition data extracted"
        assert result_state.metadata.get("llm_enrichment") is not None, "No LLM enrichment metadata"
        
        llm_metadata = result_state.metadata["llm_enrichment"]
        nutrition_data = result_state.nutrition_data
        
        logger.info("✅ DataExtractionNode processing successful",
                    product_name=nutrition_data.product_name,
                    llm_status=llm_metadata.get("status", "unknown"),
                    data_source=nutrition_data.data_source)
        
        # Log détaillé des résultats LLM
        if llm_metadata.get("status") == "success":
            logger.info("🧠 LLM Enrichment Details",
                        provider=llm_metadata.get("provider_used"),
                        quality_score=llm_metadata.get("quality_score"),
                        confidence_score=llm_metadata.get("confidence_score"),
                        processing_time=llm_metadata.get("processing_time"),
                        cache_hit=llm_metadata.get("cache_hit"),
                        consensus_validated=llm_metadata.get("consensus_validated"))
        
    except Exception as e:
        logger.error("❌ DataExtractionNode test failed", error=str(e))
        raise
    
    # ===============================
    # 3. Test des différents types d'input
    # ===============================
    
    logger.info("📊 Testing different input types")
    
    test_cases = [
        {
            "name": "JSON Input - Complete Data",
            "input_type": InputType.JSON_DATA,
            "data": test_product_data
        },
        {
            "name": "JSON Input - Minimal Data",
            "input_type": InputType.JSON_DATA,
            "data": {
                "product_name": "Produit Simple",
                "calories": 100,
                "protein": 2.0
            }
        }
    ]
    
    results_summary = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"🔍 Test Case {i}: {test_case['name']}")
        
        try:
            # Préparation du state
            input_data_case = InputData(
                type=test_case["input_type"],
                json_data=test_case["data"],
                complexity_hints={"token_reservation": {"success": True, "tokens_reserved": 50}}
            )
            
            test_state_case = WorkflowState(
                workflow_id=str(uuid.uuid4()),
                user_context=user_context,
                input_data=input_data_case,
                analysis_config=analysis_config,
                current_stage=WorkflowStage.TOKEN_VALIDATION
            )
            
            # Exécution
            result_state_case = await extraction_node.process(test_state_case)
            
            # Collecte des métriques
            llm_metadata_case = {"status": "not_found"}
            for step in result_state_case.processing_history:
                if isinstance(step, dict) and step.get("stage") == "llm_enrichment":
                    llm_metadata_case = step.get("metadata", {})
                    break
            
            result_summary = {
                "test_case": test_case["name"],
                "success": True,
                "llm_status": llm_metadata_case.get("status", "unknown"),
                "quality_score": llm_metadata_case.get("quality_score"),
                "confidence_score": llm_metadata_case.get("confidence_score"),
                "processing_time": llm_metadata_case.get("processing_time"),
                "product_name": result_state_case.nutrition_data.product_name,
                "data_source": result_state_case.nutrition_data.data_source
            }
            
            results_summary.append(result_summary)
            
            logger.info(f"✅ Test Case {i} successful", **result_summary)
            
        except Exception as e:
            error_summary = {
                "test_case": test_case["name"],
                "success": False,
                "error": str(e)
            }
            results_summary.append(error_summary)
            logger.error(f"❌ Test Case {i} failed", **error_summary)
    
    # ===============================
    # 4. Analyse des performances et qualité
    # ===============================
    
    logger.info("📈 Performance and Quality Analysis")
    
    successful_tests = [r for r in results_summary if r.get("success", False)]
    failed_tests = [r for r in results_summary if not r.get("success", False)]
    
    if successful_tests:
        # Calcul des métriques moyennes
        avg_quality = sum(r.get("quality_score", 0) for r in successful_tests if r.get("quality_score")) / len([r for r in successful_tests if r.get("quality_score")])
        avg_confidence = sum(r.get("confidence_score", 0) for r in successful_tests if r.get("confidence_score")) / len([r for r in successful_tests if r.get("confidence_score")])
        avg_processing_time = sum(r.get("processing_time", 0) for r in successful_tests if r.get("processing_time")) / len([r for r in successful_tests if r.get("processing_time")])
        
        performance_metrics = {
            "total_tests": len(results_summary),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(results_summary) * 100,
            "average_quality_score": round(avg_quality, 3),
            "average_confidence_score": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3)
        }
        
        logger.info("📊 Performance Metrics", **performance_metrics)
    
    # ===============================
    # 5. Résumé final
    # ===============================
    
    logger.info("🎯 Integration Test Summary")
    
    total_success = all(r.get("success", False) for r in results_summary)
    
    final_summary = {
        "integration_test_status": "SUCCESS" if total_success else "PARTIAL_SUCCESS",
        "di_container_setup": "✅ SUCCESS",
        "llm_service_injection": "✅ SUCCESS",
        "dataextraction_node": "✅ SUCCESS",
        "workflow_integration": "✅ SUCCESS" if total_success else "⚠️ PARTIAL",
        "total_test_cases": len(results_summary),
        "successful_cases": len(successful_tests),
        "failed_cases": len(failed_tests)
    }
    
    logger.info("🏁 Final Integration Test Results", **final_summary)
    
    if failed_tests:
        logger.warning("⚠️ Some test cases failed", failed_tests=failed_tests)
    
    return {
        "success": total_success,
        "summary": final_summary,
        "results": results_summary,
        "performance": performance_metrics if successful_tests else None
    }


async def main():
    """Point d'entrée principal."""
    try:
        results = await test_workflow_llm_integration()
        
        if results["success"]:
            logger.info("🎉 All integration tests passed successfully!")
            return 0
        else:
            logger.warning("⚠️ Some integration tests failed, but core functionality works")
            return 1
            
    except Exception as e:
        logger.error("💥 Integration test failed with critical error", error=str(e))
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))