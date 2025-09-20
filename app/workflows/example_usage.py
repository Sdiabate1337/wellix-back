"""
Exemple d'utilisation du système de workflow LangGraph.
Démontre l'orchestration complète d'analyse nutritionnelle.

Architecture Pattern : Facade + Factory + Dependency Injection
Inspiration : Spring Boot Application, .NET Core Startup, NestJS Module
"""

import asyncio
from typing import Dict, Any
from datetime import datetime
import structlog
import json

# Imports du système workflow
from app.workflows.orchestrator import LangGraphWorkflowOrchestrator
from app.workflows.container import EnhancedWorkflowContainer
from app.workflows.interfaces import InputData, InputType, AnalysisConfig, QualityLevel
from app.workflows.nodes.core_nodes import InputValidationNode, TokenValidationNode
from app.workflows.nodes.advanced_nodes import DataExtractionNode, ExpertAnalysisNode
from app.workflows.nodes.completion_nodes import (
    NutritionEnrichmentNode, HealthProfileContextNode, ScoreCalculationNode
)
from app.workflows.nodes.final_nodes import (
    AlternativeGenerationNode, ChatContextPreparationNode, ResponseAssemblyNode
)

# Imports des modèles
from app.models.health import UserHealthContext, ActivityLevel
from app.models.clinical import Gender

logger = structlog.get_logger(__name__)


class WorkflowSystemExample:
    """
    Exemple complet d'utilisation du système de workflow.
    
    Cette classe démontre :
    - Configuration et initialisation du système
    - Exécution d'analyses sur différents types d'inputs
    - Gestion des erreurs et monitoring
    - Personnalisation selon les besoins business
    """
    
    def __init__(self):
        self.container = EnhancedWorkflowContainer()
        self.orchestrator = None
        self._setup_system()
    
    def _setup_system(self) -> None:
        """Configure le système complet."""
        # Enregistrement des services dans le conteneur DI
        self._register_services()
        
        # Création de l'orchestrateur avec tous les nodes
        self.orchestrator = self._create_orchestrator()
        
        logger.info("Workflow system initialized successfully")
    
    def _register_services(self) -> None:
        """Enregistre tous les services nécessaires."""
        # Services de base
        self.container.register_singleton("logger", structlog.get_logger)
        
        # Configuration par défaut
        default_config = {
            "max_tokens_per_request": 1000,
            "enable_caching": True,
            "timeout_seconds": 300
        }
        self.container.register_singleton("config", lambda: default_config)
        
        # Services métier (mock implementations)
        self.container.register_transient("ocr_service", self._create_mock_ocr_service)
        self.container.register_transient("nutrition_api", self._create_mock_nutrition_api)
        self.container.register_transient("llm_service", self._create_mock_llm_service)
        
        logger.info("Services registered in DI container")
    
    def _create_mock_ocr_service(self):
        """Crée un service OCR mock."""
        class MockOCRService:
            async def extract_text(self, image_data: bytes) -> str:
                await asyncio.sleep(0.1)  # Simulation
                return "Mock OCR extracted text with nutrition info"
        
        return MockOCRService()
    
    def _create_mock_nutrition_api(self):
        """Crée un service API nutrition mock."""
        class MockNutritionAPI:
            async def get_by_barcode(self, barcode: str) -> Dict[str, Any]:
                await asyncio.sleep(0.05)
                return {
                    "product_name": f"Product {barcode}",
                    "brand": "Mock Brand",
                    "calories": 250,
                    "protein": 8.5,
                    "carbohydrates": 35.0,
                    "total_fat": 12.0
                }
        
        return MockNutritionAPI()
    
    def _create_mock_llm_service(self):
        """Crée un service LLM mock."""
        class MockLLMService:
            async def analyze_nutrition(self, data: Dict[str, Any]) -> Dict[str, Any]:
                await asyncio.sleep(0.2)
                return {
                    "analysis": "Mock expert analysis",
                    "recommendations": ["Reduce sugar intake", "Increase fiber"],
                    "health_impact": "Moderate nutritional value"
                }
        
        return MockLLMService()
    
    def _create_orchestrator(self) -> LangGraphWorkflowOrchestrator:
        """Crée l'orchestrateur avec tous les nodes enregistrés."""
        orchestrator = LangGraphWorkflowOrchestrator()
        
        # Enregistrement de tous les nodes du workflow
        nodes = [
            InputValidationNode(),
            TokenValidationNode(),
            DataExtractionNode(),
            NutritionEnrichmentNode(),
            HealthProfileContextNode(),
            ExpertAnalysisNode(),
            ScoreCalculationNode(),
            AlternativeGenerationNode(),
            ChatContextPreparationNode(),
            ResponseAssemblyNode()
        ]
        
        for node in nodes:
            orchestrator.register_node(node)
        
        logger.info(f"Orchestrator configured with {len(nodes)} nodes")
        return orchestrator
    
    async def run_image_analysis_example(self) -> Dict[str, Any]:
        """
        Exemple d'analyse d'une image de produit alimentaire.
        
        Cas d'usage : Utilisateur prend une photo d'un produit en magasin
        """
        logger.info("Starting image analysis example")
        
        # Données d'entrée simulées
        input_data = InputData(
            type=InputType.IMAGE,
            image_data=b"fake_image_data_for_demo",  # En réalité, bytes de l'image
            metadata={
                "source": "mobile_app",
                "image_quality": "high",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Contexte utilisateur
        user_context = UserHealthContext(
            user_id="user_123",
            age=32,
            gender=Gender.FEMALE,
            weight=65.0,
            height=170.0,
            activity_level=ActivityLevel.MODERATE,
            health_conditions=["diabetes"],
            allergies=["nuts", "dairy"],
            dietary_restrictions=["vegetarian"]
        )
        
        # Configuration d'analyse
        analysis_config = AnalysisConfig(
            quality_level=QualityLevel.PREMIUM,
            enable_alternatives=True,
            enable_chat_context=True,
            response_format="detailed"
        )
        
        # Exécution du workflow
        try:
            result = await self.orchestrator.execute(
                input_data=input_data,
                user_context=user_context,
                analysis_config=analysis_config
            )
            
            logger.info(
                "Image analysis completed",
                workflow_id=result.workflow_id,
                success=result.success,
                duration_ms=result.performance_metrics.total_duration_ms
            )
            
            return result.data
            
        except Exception as e:
            logger.error("Image analysis failed", error=str(e))
            raise
    
    async def run_barcode_analysis_example(self) -> Dict[str, Any]:
        """
        Exemple d'analyse par code-barres.
        
        Cas d'usage : Scan rapide de code-barres pour analyse express
        """
        logger.info("Starting barcode analysis example")
        
        # Données d'entrée
        input_data = InputData(
            type=InputType.BARCODE,
            barcode="3017620422003",  # Code-barres Nutella
            metadata={
                "source": "barcode_scanner",
                "scan_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Utilisateur avec hypertension
        user_context = UserHealthContext(
            user_id="user_456",
            age=45,
            gender=Gender.MALE,
            weight=80.0,
            height=175.0,
            activity_level=ActivityLevel.LIGHT,
            health_conditions=["hypertension"],
            allergies=[],
            dietary_restrictions=[]
        )
        
        # Configuration rapide
        analysis_config = AnalysisConfig(
            quality_level=QualityLevel.STANDARD,
            enable_alternatives=True,
            enable_chat_context=False,
            response_format="summary"
        )
        
        try:
            result = await self.orchestrator.execute(
                input_data=input_data,
                user_context=user_context,
                analysis_config=analysis_config
            )
            
            logger.info(
                "Barcode analysis completed",
                workflow_id=result.workflow_id,
                success=result.success
            )
            
            return result.data
            
        except Exception as e:
            logger.error("Barcode analysis failed", error=str(e))
            raise
    
    async def run_json_analysis_example(self) -> Dict[str, Any]:
        """
        Exemple d'analyse de données JSON structurées.
        
        Cas d'usage : Import de données nutritionnelles depuis une API
        """
        logger.info("Starting JSON analysis example")
        
        # Données nutritionnelles structurées
        nutrition_json = {
            "product_name": "Organic Greek Yogurt",
            "brand": "Healthy Brand",
            "nutrition_per_100g": {
                "calories": 90,
                "protein": 10.0,
                "carbohydrates": 6.0,
                "total_fat": 3.0,
                "fiber": 0.0,
                "sugar": 4.0,
                "sodium": 0.05
            },
            "ingredients": ["organic milk", "live cultures"],
            "allergens": ["milk"]
        }
        
        input_data = InputData(
            type=InputType.JSON_DATA,
            json_data=nutrition_json,
            metadata={
                "source": "nutrition_api",
                "confidence": 0.95
            }
        )
        
        # Utilisateur fitness
        user_context = UserHealthContext(
            user_id="user_789",
            age=28,
            gender=Gender.MALE,
            weight=75.0,
            height=180.0,
            activity_level=ActivityLevel.VERY_ACTIVE,
            health_conditions=[],
            allergies=[],
            dietary_restrictions=["high_protein"]
        )
        
        # Configuration expert
        analysis_config = AnalysisConfig(
            quality_level=QualityLevel.EXPERT,
            enable_alternatives=True,
            enable_chat_context=True,
            response_format="chat_ready"
        )
        
        try:
            result = await self.orchestrator.execute(
                input_data=input_data,
                user_context=user_context,
                analysis_config=analysis_config
            )
            
            logger.info(
                "JSON analysis completed",
                workflow_id=result.workflow_id,
                success=result.success
            )
            
            return result.data
            
        except Exception as e:
            logger.error("JSON analysis failed", error=str(e))
            raise
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """
        Test de performance avec analyses multiples.
        
        Démontre la capacité du système à traiter plusieurs requêtes.
        """
        logger.info("Starting performance test")
        
        # Configuration de test
        num_requests = 5
        results = []
        
        # Lancement de plusieurs analyses en parallèle
        tasks = []
        for i in range(num_requests):
            # Variation des inputs pour le test
            if i % 3 == 0:
                task = self.run_barcode_analysis_example()
            elif i % 3 == 1:
                task = self.run_json_analysis_example()
            else:
                task = self.run_image_analysis_example()
            
            tasks.append(task)
        
        # Exécution parallèle
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        # Analyse des résultats
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes
        total_time = (end_time - start_time) * 1000  # ms
        
        performance_summary = {
            "total_requests": num_requests,
            "successful_requests": successes,
            "failed_requests": failures,
            "total_time_ms": total_time,
            "average_time_per_request_ms": total_time / num_requests,
            "requests_per_second": num_requests / (total_time / 1000)
        }
        
        logger.info(
            "Performance test completed",
            **performance_summary
        )
        
        return performance_summary
    
    def print_system_architecture(self) -> None:
        """Affiche l'architecture du système pour debug."""
        print("\n" + "="*80)
        print("🏗️  WELLIX WORKFLOW SYSTEM ARCHITECTURE")
        print("="*80)
        
        print("\n📋 REGISTERED NODES:")
        if hasattr(self.orchestrator, '_nodes'):
            for stage, node in self.orchestrator._nodes.items():
                print(f"  • {stage.value:<30} → {type(node).__name__}")
        
        print("\n🔧 DI CONTAINER SERVICES:")
        print(f"  • Registered services: {len(self.container._services)}")
        print(f"  • Singleton services: {len(self.container._singletons)}")
        
        print("\n⚙️  WORKFLOW STAGES:")
        from app.workflows.interfaces import WorkflowStage
        for stage in WorkflowStage:
            print(f"  • {stage.value}")
        
        print("\n🔄 DESIGN PATTERNS USED:")
        patterns = [
            "Strategy Pattern (Interchangeable algorithms)",
            "Template Method (Common processing flow)",
            "Factory Pattern (Extensible object creation)",
            "Dependency Injection (Loose coupling)",
            "Chain of Responsibility (Validation pipeline)",
            "State Machine (Workflow transitions)",
            "Observer Pattern (Event handling)",
            "Command Pattern (Request encapsulation)"
        ]
        for pattern in patterns:
            print(f"  • {pattern}")
        
        print("\n" + "="*80)


async def main():
    """
    Fonction principale démontrant l'utilisation complète du système.
    """
    print("🚀 Démarrage de l'exemple Wellix Workflow System")
    
    # Initialisation du système
    system = WorkflowSystemExample()
    
    # Affichage de l'architecture
    system.print_system_architecture()
    
    print("\n📊 EXÉCUTION DES EXEMPLES D'ANALYSE:")
    print("-" * 50)
    
    try:
        # Exemple 1 : Analyse d'image
        print("\n1️⃣  Analyse d'image (utilisateur diabétique)...")
        image_result = await system.run_image_analysis_example()
        print(f"   ✅ Score global: {image_result.get('scores', {}).get('overall_score', 'N/A')}/100")
        print(f"   ✅ Alternatives trouvées: {len(image_result.get('alternatives', []))}")
        
        # Exemple 2 : Analyse de code-barres
        print("\n2️⃣  Analyse de code-barres (utilisateur hypertendu)...")
        barcode_result = await system.run_barcode_analysis_example()
        print(f"   ✅ Score global: {barcode_result.get('overall_score', 'N/A')}/100")
        print(f"   ✅ Alertes santé: {len(barcode_result.get('health_notes', []))}")
        
        # Exemple 3 : Analyse JSON
        print("\n3️⃣  Analyse JSON (utilisateur fitness)...")
        json_result = await system.run_json_analysis_example()
        print(f"   ✅ Score: {json_result.get('detailed_data', {}).get('scores', {}).get('overall_score', 'N/A')}/100")
        print(f"   ✅ Recommandations: {len(json_result.get('recommendations', []))}")
        
        # Test de performance
        print("\n4️⃣  Test de performance (5 requêtes parallèles)...")
        perf_result = await system.run_performance_test()
        print(f"   ✅ Requêtes réussies: {perf_result['successful_requests']}/{perf_result['total_requests']}")
        print(f"   ✅ Temps moyen: {perf_result['average_time_per_request_ms']:.1f}ms")
        print(f"   ✅ Débit: {perf_result['requests_per_second']:.1f} req/sec")
        
        print("\n🎉 TOUS LES EXEMPLES EXÉCUTÉS AVEC SUCCÈS!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        logger.error("Example execution failed", error=str(e))
        raise
    
    print("\n" + "="*80)
    print("✨ Système de workflow Wellix prêt pour production!")
    print("="*80)


if __name__ == "__main__":
    # Configuration du logging pour la démo
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        level="INFO"
    )
    
    # Exécution de la démo
    asyncio.run(main())