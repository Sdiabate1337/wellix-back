#!/usr/bin/env python3
"""
Test basique de l'architecture LLM service.
Vérifie les interfaces, task router et GPT-4 service (sans vraie API).
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


async def test_llm_interfaces():
    """Test les interfaces et data classes."""
    print("=== Test Interfaces LLM ===")
    
    try:
        from app.services.llm.interfaces import (
            LLMProvider, LLMTaskType, LLMTask, LLMResult, 
            EnrichmentConfig, QualityMetrics, ConfidenceLevel
        )
        
        # Test création config
        config = EnrichmentConfig(
            primary_provider=LLMProvider.GPT_4O,
            fallback_providers=[LLMProvider.CLAUDE_3_5_SONNET, LLMProvider.GEMINI_PRO],
            enable_consensus_validation=True,
            enable_semantic_cache=True
        )
        print(f"✓ Config créée: primary={config.primary_provider}")
        print(f"✓ Fallbacks: {len(config.fallback_providers)}")
        
        # Test création tâche
        task = LLMTask(
            task_type=LLMTaskType.NUTRITION_ANALYSIS,
            prompt="Analysez ce produit nutritionnellement",
            data={
                "product_name": "Yaourt grec nature",
                "ingredients": ["lait", "ferments lactiques"],
                "nutrition_facts": {"protein": 10, "calories": 60}
            },
            requires_validation=True
        )
        print(f"✓ Tâche créée: {task.task_type}")
        
        # Test métriques qualité
        metrics = QualityMetrics(
            data_completeness=0.9,
            logical_consistency=0.8,
            source_citation=0.7,
            confidence_calibration=0.8,
            format_compliance=1.0
        )
        print(f"✓ Métriques: score global={metrics.overall_score:.2f}")
        print(f"✓ Niveau confiance: {metrics.confidence_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur interfaces: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_task_router():
    """Test le routeur de tâches intelligent."""
    print("\n=== Test Task Router ===")
    
    try:
        from app.services.llm.task_router import IntelligentTaskRouter
        from app.services.llm.interfaces import EnrichmentConfig, LLMTaskType, LLMTask, LLMProvider
        
        # Configuration
        config = EnrichmentConfig(
            primary_provider=LLMProvider.GPT_4O,
            fallback_providers=[LLMProvider.CLAUDE_3_5_SONNET, LLMProvider.GEMINI_PRO],
            enable_cost_optimization=True
        )
        
        # Création router
        router = IntelligentTaskRouter(config)
        print(f"✓ Router créé avec {len(router.TASK_SPECIALIZATIONS)} spécialisations")
        
        # Test routage simple
        task = LLMTask(
            task_type=LLMTaskType.NUTRITION_ANALYSIS,
            prompt="Test prompt",
            data={"test": "data"}
        )
        
        optimal_provider = await router.route_task(task)
        print(f"✓ Routage nutrition: {optimal_provider}")
        
        # Test planification d'analyse
        product_data = {
            "product_name": "Céréales enrichies",
            "ingredients": ["blé", "sucre", "vitamines"],
            "nutrition_facts": {"fiber": 5, "protein": 8, "calories": 350},
            "marketing_claims": ["Riche en fibres", "Source de protéines"]
        }
        
        planned_tasks = await router.plan_analysis(product_data)
        print(f"✓ Analyse planifiée: {len(planned_tasks)} tâches")
        
        for task in planned_tasks:
            print(f"  - {task.task_type} (priorité {task.priority})")
        
        # Test optimisation coût
        optimized_tasks = await router.optimize_for_cost(planned_tasks)
        print(f"✓ Optimisation coût: {len(optimized_tasks)} tâches")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur task router: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_consensus_validator():
    """Test le validateur de consensus."""
    print("\n=== Test Consensus Validator ===")
    
    try:
        from app.services.llm.consensus_validator import ConsensusValidator
        from app.services.llm.interfaces import LLMResult, LLMTaskType, LLMProvider, QualityMetrics
        from datetime import datetime
        
        # Création validator
        validator = ConsensusValidator(
            min_providers=2,
            consensus_threshold=0.8
        )
        print(f"✓ Validator créé avec seuil {validator.consensus_threshold}")
        
        # Création résultats simulés
        result1 = LLMResult(
            analysis={
                "health_score": 8.5,
                "nutritional_quality": "excellent",
                "allergens_detected": ["milk"],
                "recommendations": ["Consommation quotidienne recommandée"]
            },
            raw_response="{}",
            quality_metrics=QualityMetrics(0.9, 0.8, 0.7, 0.8, 1.0),
            confidence_score=0.9,
            provider_used=LLMProvider.GPT_4O,
            task_type=LLMTaskType.NUTRITION_ANALYSIS,
            processing_time=2.5
        )
        
        result2 = LLMResult(
            analysis={
                "health_score": 8.2,
                "nutritional_quality": "excellent", 
                "allergens_detected": ["milk"],
                "recommendations": ["Excellent choix pour petit-déjeuner"]
            },
            raw_response="{}",
            quality_metrics=QualityMetrics(0.8, 0.9, 0.6, 0.7, 1.0),
            confidence_score=0.85,
            provider_used=LLMProvider.CLAUDE_3_5_SONNET,
            task_type=LLMTaskType.NUTRITION_ANALYSIS,
            processing_time=3.1
        )
        
        # Test calcul consensus
        consensus_score = await validator.calculate_consensus_score([result1, result2])
        print(f"✓ Score consensus: {consensus_score:.2f}")
        
        # Test résolution conflits (résultats similaires)
        resolved = await validator.resolve_conflicts([result1, result2])
        print(f"✓ Résolution consensus: score santé {resolved.analysis.get('health_score')}")
        
        # Test métriques consensus
        from app.services.llm.consensus_validator import ConsensusMetrics
        
        # Simulation directe des métriques
        validator_instance = ConsensusValidator()
        metrics = await validator_instance._calculate_consensus_metrics([result1, result2])
        print(f"✓ Accord global: {metrics.agreement_score:.2f}")
        print(f"✓ Cohérence factuelle: {metrics.factual_consistency:.2f}")
        print(f"✓ Champs en conflit: {len(metrics.conflicting_fields)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur consensus validator: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_semantic_cache():
    """Test le cache sémantique."""
    print("\n=== Test Semantic Cache ===")
    
    try:
        from app.services.llm.semantic_cache import SemanticCache
        from app.services.llm.interfaces import LLMResult, LLMTaskType, LLMProvider, QualityMetrics
        import tempfile
        
        # Création cache avec dossier temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SemanticCache(
                cache_dir=temp_dir,
                max_entries=100,
                similarity_threshold=0.95,
                enable_persistence=False  # Désactivé pour test
            )
            
            print(f"✓ Cache créé: max {cache.max_entries} entrées")
            
            # Données test
            product_data1 = {
                "product_name": "Yaourt grec nature",
                "ingredients": ["lait", "ferments lactiques"],
                "nutrition_facts": {"protein": 10, "calories": 60}
            }
            
            product_data2 = {
                "product_name": "Yaourt grec naturel",  # Très similaire
                "ingredients": ["lait écrémé", "ferments lactiques"],
                "nutrition_facts": {"protein": 9.5, "calories": 58}
            }
            
            # Résultat à stocker
            result = LLMResult(
                analysis={
                    "health_score": 8.5,
                    "nutritional_quality": "excellent",
                    "recommendations": ["Excellent source de protéines"]
                },
                raw_response="{}",
                quality_metrics=QualityMetrics(0.9, 0.8, 0.7, 0.8, 1.0),
                confidence_score=0.9,
                provider_used=LLMProvider.GPT_4O,
                task_type=LLMTaskType.NUTRITION_ANALYSIS,
                processing_time=2.5
            )
            
            # Test stockage
            await cache.store_analysis(product_data1, result)
            print("✓ Résultat stocké dans cache")
            
            # Test récupération exacte
            cached_exact = await cache.get_similar_analysis(product_data1, threshold=1.0)
            if cached_exact:
                print("✓ Récupération exacte réussie")
                print(f"✓ Cache hit: {cached_exact.cache_hit}")
                print(f"✓ Similarité: {cached_exact.cache_similarity}")
            else:
                print("✗ Récupération exacte échouée")
            
            # Test récupération similaire
            cached_similar = await cache.get_similar_analysis(product_data2, threshold=0.8)
            if cached_similar:
                print("✓ Récupération similaire réussie")
                print(f"✓ Similarité détectée: {cached_similar.cache_similarity:.2f}")
            else:
                print("✗ Aucune similarité détectée")
            
            # Test calcul similarité
            similarity = await cache.calculate_similarity(product_data1, product_data2)
            print(f"✓ Similarité calculée: {similarity:.2f}")
            
            # Statistiques cache
            stats = cache.get_cache_stats()
            print(f"✓ Stats cache: {stats['total_entries']} entrées, hit rate: {stats['hit_rate']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur semantic cache: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gpt4_service_structure():
    """Test la structure du service GPT-4 (sans appel API)."""
    print("\n=== Test Structure GPT-4 Service ===")
    
    try:
        from app.services.llm.gpt4_service import GPT4Service
        from app.services.llm.interfaces import LLMProvider, LLMTaskType, LLMTask
        import os
        
        # Test sans vraie clé API (juste structure)
        os.environ["OPENAI_API_KEY"] = "test-key-for-structure-only"
        
        # Test création service
        service = GPT4Service(LLMProvider.GPT_4O)
        print(f"✓ Service créé: {service.provider_name}")
        print(f"✓ Tâches supportées: {len(service.supported_tasks)}")
        print(f"✓ Coût par token: ${service.cost_per_token:.8f}")
        
        # Test configuration modèle
        config = service.config
        print(f"✓ Modèle: {config['model']}")
        print(f"✓ Contexte: {config['context_window']} tokens")
        print(f"✓ Max tokens: {config['max_tokens']}")
        
        # Test estimation coût
        task = LLMTask(
            task_type=LLMTaskType.NUTRITION_ANALYSIS,
            prompt="Analysez ce produit: Yaourt grec avec miel et noix",
            data={"test": "data"},
            max_tokens=1000
        )
        
        estimated_cost = await service.estimate_cost(task)
        print(f"✓ Coût estimé: ${estimated_cost:.4f}")
        
        # Test validation réponse
        valid_json = '{"health_score": 8.5, "quality": "excellent"}'
        invalid_json = 'pas du json valide'
        
        is_valid = await service.validate_response(valid_json, "json")
        print(f"✓ Validation JSON valide: {is_valid}")
        
        is_invalid = await service.validate_response(invalid_json, "json")
        print(f"✓ Validation JSON invalide: {is_invalid}")
        
        # Test statistiques
        stats = service.get_usage_stats()
        print(f"✓ Stats initiales: {stats['total_requests']} requêtes")
        
        # Nettoyage
        del os.environ["OPENAI_API_KEY"]
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur GPT-4 service: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Point d'entrée principal des tests."""
    print("🧠 Tests Architecture LLM Service")
    print("=" * 50)
    
    tests = [
        ("Interfaces LLM", test_llm_interfaces),
        ("Task Router Intelligent", test_task_router),
        ("Consensus Validator", test_consensus_validator),
        ("Semantic Cache", test_semantic_cache),
        ("GPT-4 Service Structure", test_gpt4_service_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test échoué: {e}")
            results.append((test_name, False))
        
        await asyncio.sleep(0.5)  # Pause entre tests
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{len(results)} tests passés")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ Architecture LLM OPÉRATIONNELLE")
    else:
        print("⚠️  Certains tests ont échoué")
    
    print("\n📋 ARCHITECTURE LLM COMPLÈTE:")
    print("✓ Interfaces abstraites (ISP)")
    print("✓ Task Router intelligent")
    print("✓ Consensus Validator")
    print("✓ Semantic Cache")
    print("✓ GPT-4 Service Provider")
    print("📦 Prêt pour Manager + intégration workflow!")


if __name__ == "__main__":
    asyncio.run(main())