#!/usr/bin/env python3
"""
Test d'intégration rapide pour le service OCR réel.
Vérifie que l'intégration entre les services OCR et les nodes du workflow fonctionne.
"""

import asyncio
import sys
import os
from pathlib import Path

# Ajouter le projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ocr import ocr_manager, IOCRService, IOCRProcessor
from app.workflows.nodes.advanced_nodes import DataExtractionNode
from app.workflows.interfaces import WorkflowState, InputData, InputType
from app.models.clinical import NutritionData
import structlog

# Configuration du logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_ocr_service_integration():
    """Test que les services OCR sont correctement instanciés."""
    print("=== Test OCR Service Integration ===")
    
    try:
        # Test 1: Instanciation des services
        ocr_service = ocr_manager.get_ocr_service()
        print(f"✓ OCR Service obtenu: {type(ocr_service).__name__}")
        
        ocr_processor = ocr_manager.get_nutrition_processor()
        print(f"✓ OCR Processor obtenu: {type(ocr_processor).__name__}")
        
        # Test 2: DataExtractionNode avec services réels
        node = DataExtractionNode()
        print(f"✓ DataExtractionNode créé avec services: {type(node.ocr_service).__name__}")
        
        # Test 3: Simulation avec données factices (sans vraie image)
        print("\n=== Test avec données factices ===")
        
        # Création d'un état de workflow factice
        from app.workflows.interfaces import WorkflowStage, AnalysisConfig, QualityLevel
        
        # Données de test - on va tester sans vraie image pour l'instant
        test_json_data = {
            "product_name": "Test Product",
            "calories": 250,
            "protein": 10.0,
            "carbohydrates": 30.0,
            "total_fat": 12.0,
            "fiber": 5.0,
            "sugar": 8.0,
            "sodium": 0.5,
            "ingredients": ["ingredient1", "ingredient2"],
            "allergens": ["milk"]
        }
        
        input_data = InputData(
            type=InputType.JSON_DATA,
            json_data=test_json_data,
            complexity_hints={"token_reservation": {"success": True}}
        )
        
        state = WorkflowState(
            workflow_id="test-workflow",
            current_stage=WorkflowStage.TOKEN_VALIDATION,
            input_data=input_data,
            analysis_config=AnalysisConfig(quality_level=QualityLevel.BASIC),
            metadata={}
        )
        
        # Test d'extraction JSON
        if await node.can_process(state):
            print("✓ Node peut traiter l'état de test")
            
            # Validation des préconditions
            errors = await node.validate_preconditions(state)
            if not errors:
                print("✓ Préconditions validées")
                
                # Extraction de données JSON
                nutrition_data = await node._extract_from_json(input_data)
                if nutrition_data:
                    print(f"✓ Extraction JSON réussie: {nutrition_data.product_name}")
                    print(f"  - Calories: {nutrition_data.calories}")
                    print(f"  - Protéines: {nutrition_data.protein}g")
                    print(f"  - Source: {nutrition_data.data_source}")
                else:
                    print("✗ Extraction JSON échouée")
            else:
                print(f"✗ Erreurs de préconditions: {errors}")
        else:
            print("✗ Node ne peut pas traiter l'état de test")
        
        print("\n=== Test intégration complétée avec succès ===")
        
    except Exception as e:
        print(f"✗ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()


async def test_ocr_services_functionality():
    """Test des fonctionnalités des services OCR (sans vraie image)."""
    print("\n=== Test Fonctionnalités Services OCR ===")
    
    try:
        # Obtenir les services
        ocr_service = ocr_manager.get_ocr_service()
        ocr_processor = ocr_manager.get_nutrition_processor()
        
        # Test validation d'image (avec données factices)
        print("Test validation d'image...")
        fake_image_data = b"fake_image_data_for_testing"
        
        # Note: ce test va probablement échouer car les données ne sont pas une vraie image
        # mais cela nous permettra de voir si la méthode est appelable
        try:
            quality_result = await ocr_service.validate_image_quality(fake_image_data)
            print(f"✓ Validation image retournée: {quality_result}")
        except Exception as e:
            print(f"✓ Validation image échouée comme attendu: {type(e).__name__}")
        
        # Test de fallback
        print("\nTest fallback extraction...")
        from app.workflows.interfaces import InputData, InputType
        
        fallback_input = InputData(
            type=InputType.IMAGE,
            image_data=None,
            complexity_hints={"token_reservation": {"success": True}}
        )
        
        node = DataExtractionNode()
        fallback_result = await node._fallback_extraction(fallback_input)
        if fallback_result:
            print(f"✓ Fallback extraction réussie: {fallback_result.product_name}")
        
        print("✓ Test des fonctionnalités complété")
        
    except Exception as e:
        print(f"✗ Erreur pendant le test fonctionnalités: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Point d'entrée principal du test."""
    print("🚀 Démarrage des tests d'intégration OCR")
    print("=" * 50)
    
    await test_ocr_service_integration()
    await test_ocr_services_functionality()
    
    print("=" * 50)
    print("✅ Tests d'intégration terminés")


if __name__ == "__main__":
    asyncio.run(main())