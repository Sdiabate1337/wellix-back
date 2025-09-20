"""
Factory Interfaces for Workflow Components.
Implements Factory Pattern + Abstract Factory Pattern for extensible component creation.

Architecture Pattern : Abstract Factory + Builder + Strategy
Inspiration : Spring Framework, Google Guice, Dagger 2
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, TypeVar
from enum import Enum

from app.workflows.interfaces import (
    IWorkflowNode, IExtractionStrategy, IAnalysisStrategy, IAlternativeStrategy,
    InputType, QualityLevel, AlternativeType, WorkflowStage
)
from app.models.health import UserHealthContext

T = TypeVar('T')


# ================================
# FACTORY INTERFACES
# ================================

class INodeFactory(ABC):
    """
    Abstract Factory pour la création de nodes workflow.
    
    Permet d'ajouter de nouveaux types de nodes sans modifier le code existant.
    Pattern : Abstract Factory + Registry
    """
    
    @abstractmethod
    def create_node(self, stage: WorkflowStage, **kwargs) -> IWorkflowNode:
        """
        Crée un node pour le stage donné.
        
        Args:
            stage: Stage du workflow
            **kwargs: Paramètres spécifiques au node
            
        Returns:
            Instance du node approprié
            
        Raises:
            UnsupportedStageError: Si le stage n'est pas supporté
        """
        pass
    
    @abstractmethod
    def get_supported_stages(self) -> List[WorkflowStage]:
        """
        Retourne la liste des stages supportés par cette factory.
        
        Returns:
            Liste des stages supportés
        """
        pass
    
    @abstractmethod
    def register_node_type(self, stage: WorkflowStage, node_class: Type[IWorkflowNode]) -> None:
        """
        Enregistre un nouveau type de node pour un stage.
        
        Args:
            stage: Stage du workflow
            node_class: Classe du node à enregistrer
        """
        pass


class IExtractionStrategyFactory(ABC):
    """
    Factory pour les stratégies d'extraction de données.
    
    Sélectionne la meilleure stratégie selon le type d'input et les exigences.
    Pattern : Strategy Factory + Chain of Responsibility
    """
    
    @abstractmethod
    def create_strategy(self, 
                       input_type: InputType, 
                       quality_level: QualityLevel,
                       **context) -> IExtractionStrategy:
        """
        Crée la stratégie d'extraction optimale.
        
        Args:
            input_type: Type d'input (image, barcode, JSON)
            quality_level: Niveau de qualité requis
            **context: Contexte additionnel (user preferences, etc.)
            
        Returns:
            Stratégie d'extraction appropriée
        """
        pass
    
    @abstractmethod
    def get_available_strategies(self, input_type: InputType) -> List[Type[IExtractionStrategy]]:
        """
        Retourne les stratégies disponibles pour un type d'input.
        
        Args:
            input_type: Type d'input
            
        Returns:
            Liste des stratégies disponibles
        """
        pass


class IAnalysisStrategyFactory(ABC):
    """
    Factory pour les stratégies d'analyse de santé.
    
    Sélectionne la stratégie d'analyse selon le profil utilisateur et le niveau de qualité.
    Pattern : Strategy Factory + Builder
    """
    
    @abstractmethod
    def create_strategy(self,
                       user_context: UserHealthContext,
                       quality_level: QualityLevel,
                       **preferences) -> IAnalysisStrategy:
        """
        Crée la stratégie d'analyse personnalisée.
        
        Args:
            user_context: Contexte santé de l'utilisateur
            quality_level: Niveau de qualité d'analyse
            **preferences: Préférences additionnelles
            
        Returns:
            Stratégie d'analyse personnalisée
        """
        pass
    
    @abstractmethod
    def create_hybrid_strategy(self,
                              primary_strategy: IAnalysisStrategy,
                              fallback_strategies: List[IAnalysisStrategy]) -> IAnalysisStrategy:
        """
        Crée une stratégie hybride avec fallbacks.
        
        Args:
            primary_strategy: Stratégie principale
            fallback_strategies: Stratégies de fallback
            
        Returns:
            Stratégie hybride
        """
        pass


class IAlternativeStrategyFactory(ABC):
    """
    Factory pour les stratégies de génération d'alternatives.
    
    Crée des stratégies pour différents types d'alternatives (santé, prix, proximité).
    Pattern : Strategy Factory + Composite
    """
    
    @abstractmethod
    def create_strategy(self, alternative_type: AlternativeType) -> IAlternativeStrategy:
        """
        Crée une stratégie pour un type d'alternative.
        
        Args:
            alternative_type: Type d'alternative requis
            
        Returns:
            Stratégie d'alternative appropriée
        """
        pass
    
    @abstractmethod
    def create_composite_strategy(self, 
                                 alternative_types: List[AlternativeType],
                                 weights: Optional[Dict[AlternativeType, float]] = None) -> IAlternativeStrategy:
        """
        Crée une stratégie composite pour plusieurs types d'alternatives.
        
        Args:
            alternative_types: Types d'alternatives à combiner
            weights: Poids pour chaque type (optionnel)
            
        Returns:
            Stratégie composite
        """
        pass


# ================================
# CONFIGURATION INTERFACES
# ================================

class IWorkflowConfiguration(ABC):
    """
    Interface pour la configuration du workflow.
    
    Centralise la configuration et permet les feature flags.
    Pattern : Configuration + Strategy
    """
    
    @abstractmethod
    def get_node_timeout(self, stage: WorkflowStage) -> float:
        """Timeout en secondes pour un stage."""
        pass
    
    @abstractmethod
    def get_retry_config(self, stage: WorkflowStage) -> Dict[str, Any]:
        """Configuration de retry pour un stage."""
        pass
    
    @abstractmethod
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Vérifie si une feature est activée."""
        pass
    
    @abstractmethod
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Seuils de qualité pour différentes métriques."""
        pass
    
    @abstractmethod
    def get_cost_limits(self) -> Dict[str, int]:
        """Limites de coût en tokens pour différentes opérations."""
        pass


class IServiceLocator(ABC):
    """
    Interface pour la localisation de services.
    
    Pattern : Service Locator + Registry
    """
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """
        Récupère un service par son type.
        
        Args:
            service_type: Type du service
            
        Returns:
            Instance du service
        """
        pass
    
    @abstractmethod
    def register_service(self, service_type: Type[T], instance: T) -> None:
        """
        Enregistre un service.
        
        Args:
            service_type: Type du service
            instance: Instance du service
        """
        pass
    
    @abstractmethod
    def is_service_registered(self, service_type: Type[T]) -> bool:
        """
        Vérifie si un service est enregistré.
        
        Args:
            service_type: Type du service
            
        Returns:
            True si le service est enregistré
        """
        pass


# ================================
# BUILDER INTERFACES
# ================================

class IWorkflowBuilder(ABC):
    """
    Builder pour la construction de workflows complexes.
    
    Pattern : Builder + Fluent Interface
    """
    
    @abstractmethod
    def with_input_validation(self, enabled: bool = True) -> 'IWorkflowBuilder':
        """Active/désactive la validation d'input."""
        pass
    
    @abstractmethod
    def with_token_validation(self, enabled: bool = True) -> 'IWorkflowBuilder':
        """Active/désactive la validation de tokens."""
        pass
    
    @abstractmethod
    def with_quality_level(self, level: QualityLevel) -> 'IWorkflowBuilder':
        """Définit le niveau de qualité."""
        pass
    
    @abstractmethod
    def with_alternatives(self, types: List[AlternativeType]) -> 'IWorkflowBuilder':
        """Définit les types d'alternatives à générer."""
        pass
    
    @abstractmethod
    def with_chat_context(self, enabled: bool = True) -> 'IWorkflowBuilder':
        """Active/désactive la préparation du contexte chat."""
        pass
    
    @abstractmethod
    def with_custom_node(self, stage: WorkflowStage, node: IWorkflowNode) -> 'IWorkflowBuilder':
        """Ajoute un node personnalisé."""
        pass
    
    @abstractmethod
    def build(self) -> 'IWorkflowOrchestrator':
        """
        Construit l'orchestrateur de workflow.
        
        Returns:
            Orchestrateur configuré
        """
        pass


# ================================
# SPECIALIZED FACTORIES
# ================================

class IContextAwareFactory(ABC):
    """
    Factory contextuelle qui adapte ses créations selon le contexte.
    
    Pattern : Context-Aware Factory + Adaptive Strategy
    """
    
    @abstractmethod
    def create_for_context(self, 
                          context: UserHealthContext,
                          preferences: Dict[str, Any],
                          **kwargs) -> Any:
        """
        Crée un objet adapté au contexte utilisateur.
        
        Args:
            context: Contexte utilisateur
            preferences: Préférences utilisateur
            **kwargs: Paramètres additionnels
            
        Returns:
            Objet adapté au contexte
        """
        pass
    
    @abstractmethod
    def analyze_context(self, context: UserHealthContext) -> Dict[str, Any]:
        """
        Analyse le contexte pour déterminer les besoins.
        
        Args:
            context: Contexte à analyser
            
        Returns:
            Analyse du contexte
        """
        pass


class ICachingFactory(ABC):
    """
    Factory avec capacités de cache pour optimiser les performances.
    
    Pattern : Caching Factory + Proxy
    """
    
    @abstractmethod
    def create_cached(self, cache_key: str, factory_func: callable, ttl_seconds: int = 3600) -> Any:
        """
        Crée un objet avec mise en cache.
        
        Args:
            cache_key: Clé de cache
            factory_func: Fonction de création
            ttl_seconds: Durée de vie du cache
            
        Returns:
            Objet (depuis le cache ou nouvellement créé)
        """
        pass
    
    @abstractmethod
    def invalidate_cache(self, pattern: str) -> int:
        """
        Invalide les entrées de cache correspondant au pattern.
        
        Args:
            pattern: Pattern de clés à invalider
            
        Returns:
            Nombre d'entrées invalidées
        """
        pass


# ================================
# EXCEPTIONS
# ================================

class UnsupportedStageError(Exception):
    """Exception levée quand un stage n'est pas supporté par une factory."""
    
    def __init__(self, stage: WorkflowStage, factory_type: str):
        self.stage = stage
        self.factory_type = factory_type
        super().__init__(f"Stage '{stage.value}' not supported by {factory_type}")


class StrategyCreationError(Exception):
    """Exception levée quand la création d'une stratégie échoue."""
    
    def __init__(self, strategy_type: str, reason: str):
        self.strategy_type = strategy_type
        self.reason = reason
        super().__init__(f"Failed to create {strategy_type} strategy: {reason}")


class ConfigurationError(Exception):
    """Exception levée quand la configuration est invalide."""
    
    def __init__(self, config_key: str, reason: str):
        self.config_key = config_key
        self.reason = reason
        super().__init__(f"Configuration error for '{config_key}': {reason}")