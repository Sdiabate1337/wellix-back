#!/usr/bin/env python3
"""
Test d'intégration du workflow complet avec enrichissement LLM - Version simplifiée.
"""

import asyncio
import structlog
import uuid

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


async def test_workflow_integration():
    """Test d'intégration simplifié du workflow avec LLM."""
    
    logger.info("🧪 Starting Simplified Workflow LLM Integration Test")
    
    try:
        # Setup du DI Container
        logger.info("📦 Setting up DI Container with LLM services")
        from app.workflows.container import workflow_container, register_llm_services
        
        register_llm_services()
        
        from app.services.llm.interfaces import ILLMEnricher
        assert workflow_container.is_registered(ILLMEnricher), "ILLMEnricher not registered"
        
        logger.info("✅ DI Container setup successful")
        
        # Test DataExtractionNode
        logger.info("🔄 Testing DataExtractionNode with LLM enrichment")
        
        from app.workflows.nodes.advanced_nodes import DataExtractionNode
        from app.workflows.interfaces import WorkflowState, WorkflowStage, InputData, InputType, AnalysisConfig, QualityLevel
        from app.models.health import UserHealthContext
        
        # Création du node
        extraction_node = DataExtractionNode()
        assert extraction_node.llm_enricher is not None, "LLM enricher not injected"
        
        # Données de test
        test_data = {
            "product_name": "Yaourt Bio Nature",
            "calories": 80,
            "protein": 4.5,
            "ingredients": ["lait bio", "ferments lactiques"],
            "allergens": ["lait"]
        }
        
        # Objets requis
        user_context = UserHealthContext(
            user_id="test_user_123",
            age=30,
            age_group="adult",
            weight=70.0,
            height=175.0,
            activity_level="moderately_active",
            health_conditions=[],
            dietary_restrictions=[],
            nutritional_goals=[]
        )
        
        analysis_config = AnalysisConfig(
            quality_level=QualityLevel.STANDARD,
            enable_alternatives=True,
            max_alternatives_per_type=3
        )
        
        input_data = InputData(
            type=InputType.JSON_DATA,
            json_data=test_data,
            complexity_hints={"token_reservation": {"success": True, "tokens_reserved": 100}}
        )
        
        test_state = WorkflowState(
            workflow_id=str(uuid.uuid4()),
            user_context=user_context,
            input_data=input_data,
            analysis_config=analysis_config,
            current_stage=WorkflowStage.TOKEN_VALIDATION
        )
        
        # Exécution
        logger.info("🚀 Executing DataExtractionNode")
        result_state = await extraction_node.process(test_state)
        
        # Vérifications
        assert result_state.nutrition_data is not None, "No nutrition data extracted"
        
        # Recherche des métadonnées LLM
        llm_metadata = {"status": "not_found"}
        for step in result_state.processing_history:
            if isinstance(step, dict) and step.get("stage") == "llm_enrichment":
                llm_metadata = step.get("metadata", {})
                break
        
        logger.info("✅ DataExtractionNode processing successful",
                    product_name=result_state.nutrition_data.product_name,
                    llm_status=llm_metadata.get("status", "unknown"),
                    data_source=result_state.nutrition_data.data_source)
        
        # Résumé final
        success = result_state.nutrition_data is not None
        
        final_summary = {
            "integration_test_status": "SUCCESS" if success else "FAILED",
            "di_container_setup": "✅ SUCCESS",
            "llm_service_injection": "✅ SUCCESS",
            "dataextraction_node": "✅ SUCCESS" if success else "❌ FAILED",
            "llm_enrichment_status": llm_metadata.get("status", "unknown")
        }
        
        logger.info("🏁 Final Integration Test Results", **final_summary)
        
        return success
        
    except Exception as e:
        logger.error("💥 Integration test failed", error=str(e))
        return False


async def main():
    """Point d'entrée principal."""
    try:
        success = await test_workflow_integration()
        
        if success:
            logger.info("🎉 Integration test passed successfully!")
            return 0
        else:
            logger.warning("⚠️ Integration test failed")
            return 1
            
    except Exception as e:
        logger.error("💥 Test runner failed", error=str(e))
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))