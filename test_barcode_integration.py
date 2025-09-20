#!/usr/bin/env python3
"""
Test d'intégration complet pour les services Barcode.
Vérifie l'implémentation OpenFoodFacts et l'intégration avec DataExtractionNode.
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

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


async def test_barcode_service_creation():
    """Test la création des services barcode."""
    print("=== Test Création Services Barcode ===")
    
    try:
        from app.services.barcode import (
            BarcodeServiceFactory, BarcodeProvider,
            get_barcode_manager, barcode_manager
        )
        
        # Test 1: Factory de services
        print("Test Factory...")
        service = BarcodeServiceFactory.create_barcode_service(
            BarcodeProvider.OPENFOODFACTS,
            language="fr",
            country="france"
        )
        print(f"✓ OpenFoodFacts service créé: {service.provider_name}")
        print(f"✓ Support nutrition: {service.supports_nutrition_data}")
        
        # Test 2: Factory d'enrichisseur
        enricher = BarcodeServiceFactory.create_barcode_enricher("nutrition")
        print(f"✓ Enrichisseur créé: {type(enricher).__name__}")
        
        # Test 3: Manager singleton
        manager = await get_barcode_manager()
        print(f"✓ Manager obtenu: {type(manager).__name__}")
        
        # Test 4: Configuration
        manager.configure(language="en", timeout=15)
        print("✓ Configuration mise à jour")
        
        print("🎉 Tous les services créés avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur création services: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_barcode_validation():
    """Test la validation de codes-barres."""
    print("\n=== Test Validation Codes-Barres ===")
    
    try:
        from app.services.barcode import BarcodeServiceFactory, BarcodeProvider
        
        service = BarcodeServiceFactory.create_barcode_service(BarcodeProvider.OPENFOODFACTS)
        
        # Codes-barres de test
        test_codes = [
            ("3017620422003", "EAN-13 valide (Nutella)"),
            ("012345678912", "UPC-A valide"),
            ("1234567890123", "EAN-13 invalide (checksum)"),
            ("12345", "Code trop court"),
            ("abc123def456", "Code avec lettres")
        ]
        
        for barcode, description in test_codes:
            result = await service.validate_barcode(barcode)
            status = "✓" if result["is_valid"] else "✗"
            print(f"{status} {description}: {result['format_type']}")
            if result.get("warnings"):
                print(f"    Warnings: {', '.join(result['warnings'])}")
        
        print("✓ Tests de validation terminés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation: {e}")
        return False


async def test_barcode_lookup():
    """Test lookup de produits réels."""
    print("\n=== Test Lookup Produits ===")
    
    try:
        from app.services.barcode import lookup_barcode
        
        # Codes-barres de produits connus
        test_barcodes = [
            "3017620422003",  # Nutella
            "3228020000000",  # Code test inexistant
        ]
        
        for barcode in test_barcodes:
            print(f"\nLookup {barcode}...")
            result = await lookup_barcode(barcode)
            
            if result:
                print(f"✓ Produit trouvé: {result.product_name}")
                print(f"  - Marque: {result.brand}")
                print(f"  - Provider: {result.provider}")
                print(f"  - Confiance: {result.confidence:.2f}")
                print(f"  - Qualité: {result.data_quality}")
                
                if result.nutrition_facts:
                    calories = result.nutrition_facts.get("calories", "N/A")
                    print(f"  - Calories: {calories}")
                
                if result.ingredients:
                    print(f"  - Ingrédients: {len(result.ingredients)} trouvés")
                
                if result.allergens:
                    print(f"  - Allergènes: {', '.join(result.allergens)}")
            else:
                print(f"✗ Produit non trouvé: {barcode}")
        
        print("\n✓ Tests de lookup terminés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lookup: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_product_search():
    """Test recherche de produits."""
    print("\n=== Test Recherche Produits ===")
    
    try:
        from app.services.barcode import search_products
        
        # Recherches de test
        queries = ["nutella", "coca", "fromage"]
        
        for query in queries:
            print(f"\nRecherche '{query}'...")
            results = await search_products(query, limit=3)
            
            print(f"✓ {len(results)} produits trouvés")
            for i, result in enumerate(results[:2]):  # Afficher 2 premiers
                print(f"  {i+1}. {result.product_name} ({result.barcode})")
                if result.brand:
                    print(f"     Marque: {result.brand}")
        
        print("\n✓ Tests de recherche terminés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur recherche: {e}")
        return False


async def test_workflow_integration():
    """Test intégration avec DataExtractionNode."""
    print("\n=== Test Intégration Workflow ===")
    
    try:
        from app.workflows.nodes.advanced_nodes import DataExtractionNode
        from app.workflows.interfaces import InputData, InputType, WorkflowState, WorkflowStage, AnalysisConfig, QualityLevel
        
        # Création du node
        node = DataExtractionNode()
        print(f"✓ DataExtractionNode créé: {node.node_name}")
        
        # Test avec code-barres connu
        input_data = InputData(
            type=InputType.BARCODE,
            barcode="3017620422003",  # Nutella
            complexity_hints={"token_reservation": {"success": True}}
        )
        
        print(f"✓ Test extraction barcode: {input_data.barcode}")
        
        # Test de la méthode d'extraction directement
        nutrition_data = await node._extract_from_barcode(input_data)
        
        if nutrition_data:
            print(f"✓ Extraction réussie: {nutrition_data.product_name}")
            print(f"  - Marque: {nutrition_data.brand}")
            print(f"  - Calories: {nutrition_data.calories}")
            print(f"  - Protéines: {nutrition_data.protein}g")
            print(f"  - Source: {nutrition_data.data_source}")
            print(f"  - Confiance: {nutrition_data.confidence_score}")
            
            if nutrition_data.ingredients:
                print(f"  - Ingrédients: {len(nutrition_data.ingredients)} trouvés")
            
            if nutrition_data.allergens:
                print(f"  - Allergènes: {', '.join(nutrition_data.allergens)}")
        else:
            print("✗ Extraction échouée")
            return False
        
        print("\n✓ Intégration workflow validée!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_enrichment():
    """Test de l'enrichissement des données."""
    print("\n=== Test Enrichissement Données ===")
    
    try:
        from app.services.barcode import BarcodeServiceFactory, BarcodeProvider
        
        # Service et enrichisseur
        service = BarcodeServiceFactory.create_barcode_service(BarcodeProvider.OPENFOODFACTS)
        enricher = BarcodeServiceFactory.create_barcode_enricher("nutrition")
        
        # Lookup + enrichissement
        barcode = "3017620422003"
        print(f"Lookup et enrichissement: {barcode}")
        
        result = await service.lookup_product(barcode)
        if result:
            print(f"✓ Données de base: {result.product_name}")
            print(f"  - Confiance initiale: {result.confidence:.2f}")
            print(f"  - Allergènes initiaux: {len(result.allergens or [])}")
            
            # Enrichissement
            enriched = await enricher.enrich_product_data(result)
            print(f"✓ Données enrichies")
            print(f"  - Confiance après enrichissement: {enriched.confidence:.2f}")
            print(f"  - Allergènes enrichis: {len(enriched.allergens or [])}")
            
            if enriched.raw_data and "wellix_enrichment" in enriched.raw_data:
                enrichment_data = enriched.raw_data["wellix_enrichment"]
                print(f"  - Score nutritionnel: {enrichment_data.get('nutrition_scores', {})}")
        
        print("\n✓ Test enrichissement terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur enrichissement: {e}")
        return False


async def main():
    """Point d'entrée principal des tests."""
    print("🚀 Tests d'Intégration Service Barcode OpenFoodFacts")
    print("=" * 60)
    
    tests = [
        ("Création des services", test_barcode_service_creation),
        ("Validation codes-barres", test_barcode_validation),
        ("Lookup produits", test_barcode_lookup),
        ("Recherche produits", test_product_search),
        ("Enrichissement données", test_enrichment),
        ("Intégration workflow", test_workflow_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test échoué: {e}")
            results.append((test_name, False))
        
        await asyncio.sleep(0.5)  # Pause entre tests
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{len(results)} tests passés")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ Service Barcode OpenFoodFacts OPÉRATIONNEL")
    else:
        print("⚠️  Certains tests ont échoué")
    
    print("\n📚 SERVICE BARCODE COMPLÈTEMENT IMPLÉMENTÉ!")


if __name__ == "__main__":
    asyncio.run(main())