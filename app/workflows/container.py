"""
Enhanced Dependency Injection Container pour le workflow.
Implémente l'Inversion of Control selon les principes SOLID + Enterprise patterns.

Architecture Pattern : IoC Container + Factory + Registry + Lifecycle Management
Inspiration : Spring IoC, Google Guice, Microsoft .NET DI, NestJS
"""

from typing import Dict, Type, TypeVar, Any, Callable, Optional, List, Union
from abc import ABC, abstractmethod
import structlog
from enum import Enum
import asyncio
import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
import weakref

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Durée de vie des services - Lifecycle Management Pattern."""
    SINGLETON = "singleton"    # Une seule instance partagée
    TRANSIENT = "transient"    # Nouvelle instance à chaque résolution  
    SCOPED = "scoped"         # Une instance par scope (request, workflow, etc.)
    LAZY_SINGLETON = "lazy_singleton"  # Singleton créé lors du premier accès


@dataclass
class ServiceDescriptor:
    """Descripteur complet d'un service enregistré."""
    interface: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None  # Pour les singletons pré-créés
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None
    
    def __post_init__(self):
        """Validation du descripteur."""
        if not self.implementation and not self.factory and not self.instance:
            raise ValueError("ServiceDescriptor must have implementation, factory, or instance")
        
        if self.implementation and self.factory:
            raise ValueError("ServiceDescriptor cannot have both implementation and factory")


@dataclass 
class ResolutionContext:
    """Contexte de résolution pour éviter les dépendances circulaires."""
    resolving_stack: List[Type] = field(default_factory=list)
    scope_instances: Dict[Type, Any] = field(default_factory=dict)
    
    def is_resolving(self, service_type: Type) -> bool:
        """Vérifie si un service est en cours de résolution (détection cycle)."""
        return service_type in self.resolving_stack
    
    def push_resolving(self, service_type: Type) -> None:
        """Ajoute un service à la pile de résolution."""
        self.resolving_stack.append(service_type)
    
    def pop_resolving(self) -> Optional[Type]:
        """Retire le dernier service de la pile de résolution."""
        return self.resolving_stack.pop() if self.resolving_stack else None


class IServiceContainer(ABC):
    """Interface pour le conteneur de services - Dependency Inversion."""
    
    @abstractmethod
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Enregistre un service comme singleton."""
        pass
    
    @abstractmethod
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Enregistre un service comme transient."""
        pass
    
    @abstractmethod
    def register_scoped(self, interface: Type[T], implementation: Type[T]) -> None:
        """Enregistre un service comme scoped."""
        pass
    
    @abstractmethod
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Enregistre une factory pour un service."""
        pass
    
    @abstractmethod
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Enregistre une instance pré-créée."""
        pass
    
    @abstractmethod
    def resolve(self, interface: Type[T]) -> T:
        """Résout une instance du service."""
        pass
    
    @abstractmethod
    async def resolve_async(self, interface: Type[T]) -> T:
        """Résout une instance du service (version async)."""
        pass
    
    @abstractmethod
    def is_registered(self, interface: Type[T]) -> bool:
        """Vérifie si un service est enregistré."""
        pass
    
    @abstractmethod
    def create_scope(self) -> 'IServiceScope':
        """Crée un nouveau scope pour les services scoped."""
        pass


class IServiceScope(ABC):
    """Interface pour un scope de services."""
    
    @abstractmethod
    def resolve(self, interface: Type[T]) -> T:
        """Résout un service dans ce scope."""
        pass
    
    @abstractmethod
    async def resolve_async(self, interface: Type[T]) -> T:
        """Résout un service dans ce scope (async)."""
        pass
    
    @abstractmethod
    def dispose(self) -> None:
        """Libère les ressources du scope."""
        pass
    
    @abstractmethod
    async def dispose_async(self) -> None:
        """Libère les ressources du scope (async)."""
        pass


class EnhancedWorkflowContainer(IServiceContainer):
    """
    Conteneur d'injection de dépendances avancé pour le workflow.
    
    Features :
    - Lifecycle management (Singleton, Transient, Scoped)
    - Dependency injection automatique
    - Circular dependency detection
    - Async support
    - Performance monitoring
    - Thread safety
    """
    
    def __init__(self):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._singleton_instances: Dict[Type, Any] = {}
        self._lock = Lock()  # Thread safety
        self._performance_stats: Dict[str, Any] = {}
        self._logger = logger.bind(component="EnhancedDIContainer")
        
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Enregistre un service comme singleton."""
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            dependencies=self._extract_dependencies(implementation)
        )
        
        with self._lock:
            self._descriptors[interface] = descriptor
            
        self._logger.info(
            "Registered singleton service",
            interface=interface.__name__,
            implementation=implementation.__name__,
            dependencies=[dep.__name__ for dep in descriptor.dependencies]
        )
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Enregistre un service comme transient."""
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            dependencies=self._extract_dependencies(implementation)
        )
        
        with self._lock:
            self._descriptors[interface] = descriptor
            
        self._logger.info(
            "Registered transient service",
            interface=interface.__name__,
            implementation=implementation.__name__
        )
    
    def register_scoped(self, interface: Type[T], implementation: Type[T]) -> None:
        """Enregistre un service comme scoped."""
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED,
            dependencies=self._extract_dependencies(implementation)
        )
        
        with self._lock:
            self._descriptors[interface] = descriptor
            
        self._logger.info(
            "Registered scoped service",
            interface=interface.__name__,
            implementation=implementation.__name__
        )
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Enregistre une factory pour créer des instances."""
        descriptor = ServiceDescriptor(
            interface=interface,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT
        )
        
        with self._lock:
            self._descriptors[interface] = descriptor
            
        self._logger.info(
            "Registered factory service",
            interface=interface.__name__
        )
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Enregistre une instance pré-créée comme singleton."""
        descriptor = ServiceDescriptor(
            interface=interface,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        
        with self._lock:
            self._descriptors[interface] = descriptor
            self._singleton_instances[interface] = instance
            
        self._logger.info(
            "Registered instance service",
            interface=interface.__name__,
            instance_type=type(instance).__name__
        )
    
    def resolve(self, interface: Type[T]) -> T:
        """Résout une instance du service de manière synchrone."""
        context = ResolutionContext()
        return self._resolve_with_context(interface, context)
    
    async def resolve_async(self, interface: Type[T]) -> T:
        """Résout une instance du service de manière asynchrone."""
        context = ResolutionContext()
        return await self._resolve_async_with_context(interface, context)
    
    def _resolve_with_context(self, interface: Type[T], context: ResolutionContext) -> T:
        """Résolution synchrone avec contexte."""
        if not self.is_registered(interface):
            raise ServiceNotRegisteredError(f"Service {interface.__name__} is not registered")
        
        # Détection de dépendances circulaires
        if context.is_resolving(interface):
            cycle_path = " -> ".join([t.__name__ for t in context.resolving_stack + [interface]])
            raise CircularDependencyError(f"Circular dependency detected: {cycle_path}")
        
        descriptor = self._descriptors[interface]
        
        try:
            context.push_resolving(interface)
            
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                return self._resolve_singleton(interface, descriptor, context)
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                return self._resolve_scoped(interface, descriptor, context)
            else:  # TRANSIENT
                return self._resolve_transient(descriptor, context)
                
        except Exception as e:
            self._logger.error(
                "Failed to resolve service",
                interface=interface.__name__,
                error=str(e)
            )
            raise ServiceResolutionError(f"Failed to resolve {interface.__name__}: {e}") from e
        finally:
            context.pop_resolving()
    
    async def _resolve_async_with_context(self, interface: Type[T], context: ResolutionContext) -> T:
        """Résolution asynchrone avec contexte."""
        # Pour l'instant, délègue à la version synchrone
        # TODO: Implémenter la création asynchrone des services
        return self._resolve_with_context(interface, context)
    
    def _resolve_singleton(self, interface: Type[T], descriptor: ServiceDescriptor, context: ResolutionContext) -> T:
        """Résout un singleton - thread-safe."""
        if interface in self._singleton_instances:
            return self._singleton_instances[interface]
        
        with self._lock:
            # Double-check locking pattern
            if interface in self._singleton_instances:
                return self._singleton_instances[interface]
            
            instance = self._create_instance(descriptor, context)
            self._singleton_instances[interface] = instance
            
            self._logger.debug(
                "Created singleton instance",
                interface=interface.__name__
            )
            
            return instance
    
    def _resolve_scoped(self, interface: Type[T], descriptor: ServiceDescriptor, context: ResolutionContext) -> T:
        """Résout un service scoped."""
        if interface in context.scope_instances:
            return context.scope_instances[interface]
        
        instance = self._create_instance(descriptor, context)
        context.scope_instances[interface] = instance
        
        self._logger.debug(
            "Created scoped instance",
            interface=interface.__name__
        )
        
        return instance
    
    def _resolve_transient(self, descriptor: ServiceDescriptor, context: ResolutionContext) -> T:
        """Résout un transient - nouvelle instance."""
        instance = self._create_instance(descriptor, context)
        
        self._logger.debug(
            "Created transient instance",
            interface=descriptor.interface.__name__
        )
        
        return instance
    
    def _create_instance(self, descriptor: ServiceDescriptor, context: ResolutionContext) -> Any:
        """Crée une instance basée sur le descripteur."""
        if descriptor.instance is not None:
            return descriptor.instance
        elif descriptor.factory:
            return descriptor.factory()
        elif descriptor.implementation:
            return self._create_with_dependencies(descriptor.implementation, context)
        else:
            raise ServiceResolutionError("No creation method available")
    
    def _create_with_dependencies(self, implementation_type: Type, context: ResolutionContext) -> Any:
        """Crée une instance en résolvant automatiquement les dépendances."""
        try:
            # Inspection du constructeur
            signature = inspect.signature(implementation_type.__init__)
            parameters = signature.parameters
            
            # Skip 'self' parameter
            param_names = list(parameters.keys())[1:]
            
            if not param_names:
                # Constructeur sans paramètres
                return implementation_type()
            
            # Résolution des dépendances
            kwargs = {}
            for param_name in param_names:
                param = parameters[param_name]
                
                if param.annotation == inspect.Parameter.empty:
                    raise ServiceResolutionError(
                        f"Parameter '{param_name}' in {implementation_type.__name__} has no type annotation"
                    )
                
                # Résolution récursive de la dépendance
                dependency_instance = self._resolve_with_context(param.annotation, context)
                kwargs[param_name] = dependency_instance
            
            return implementation_type(**kwargs)
            
        except Exception as e:
            self._logger.error(
                "Failed to create instance with dependencies",
                implementation=implementation_type.__name__,
                error=str(e)
            )
            raise ServiceResolutionError(f"Failed to create {implementation_type.__name__}: {e}") from e
    
    def _extract_dependencies(self, implementation_type: Type) -> List[Type]:
        """Extrait les dépendances d'un type basé sur son constructeur."""
        try:
            signature = inspect.signature(implementation_type.__init__)
            parameters = signature.parameters
            
            dependencies = []
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                    
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
            
            return dependencies
            
        except Exception as e:
            self._logger.warning(
                "Failed to extract dependencies",
                implementation=implementation_type.__name__,
                error=str(e)
            )
            return []
    
    def is_registered(self, interface: Type[T]) -> bool:
        """Vérifie si un service est enregistré."""
        return interface in self._descriptors
    
    def create_scope(self) -> 'ServiceScope':
        """Crée un nouveau scope pour les services scoped."""
        return ServiceScope(self)
    
    def clear(self) -> None:
        """Nettoie le conteneur - utile pour les tests."""
        with self._lock:
            self._descriptors.clear()
            self._singleton_instances.clear()
            self._performance_stats.clear()
        self._logger.info("Container cleared")
    
    def get_service_info(self, interface: Type[T]) -> Optional[Dict[str, Any]]:
        """Retourne les informations sur un service enregistré."""
        if interface not in self._descriptors:
            return None
        
        descriptor = self._descriptors[interface]
        return {
            "interface": interface.__name__,
            "implementation": descriptor.implementation.__name__ if descriptor.implementation else None,
            "lifetime": descriptor.lifetime.value,
            "has_factory": descriptor.factory is not None,
            "has_instance": descriptor.instance is not None,
            "dependencies": [dep.__name__ for dep in descriptor.dependencies],
            "is_singleton_created": interface in self._singleton_instances
        }
    
    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Retourne la liste de tous les services enregistrés."""
        return {
            interface.__name__: self.get_service_info(interface)
            for interface in self._descriptors
        }


class ServiceScope(IServiceScope):
    """Scope pour les services avec lifetime SCOPED."""
    
    def __init__(self, container: EnhancedWorkflowContainer):
        self._container = container
        self._scope_instances: Dict[Type, Any] = {}
        self._disposed = False
    
    def resolve(self, interface: Type[T]) -> T:
        """Résout un service dans ce scope."""
        if self._disposed:
            raise ObjectDisposedError("ServiceScope has been disposed")
        
        context = ResolutionContext(scope_instances=self._scope_instances)
        return self._container._resolve_with_context(interface, context)
    
    async def resolve_async(self, interface: Type[T]) -> T:
        """Résout un service dans ce scope (async)."""
        if self._disposed:
            raise ObjectDisposedError("ServiceScope has been disposed")
        
        context = ResolutionContext(scope_instances=self._scope_instances)
        return await self._container._resolve_async_with_context(interface, context)
    
    def dispose(self) -> None:
        """Libère les ressources du scope."""
        if self._disposed:
            return
        
        # Dispose des instances qui implémentent IDisposable
        for instance in self._scope_instances.values():
            if hasattr(instance, 'dispose'):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing instance: {e}")
        
        self._scope_instances.clear()
        self._disposed = True
    
    async def dispose_async(self) -> None:
        """Libère les ressources du scope (async)."""
        if self._disposed:
            return
        
        # Dispose des instances qui implémentent IAsyncDisposable
        for instance in self._scope_instances.values():
            if hasattr(instance, 'dispose_async'):
                try:
                    await instance.dispose_async()
                except Exception as e:
                    logger.warning(f"Error disposing instance async: {e}")
            elif hasattr(instance, 'dispose'):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing instance: {e}")
        
        self._scope_instances.clear()
        self._disposed = True


# ================================
# EXCEPTIONS AVANCÉES
# ================================

class ServiceNotRegisteredError(Exception):
    """Exception levée quand un service n'est pas enregistré."""
    pass


class ServiceResolutionError(Exception):
    """Exception levée quand la résolution d'un service échoue."""
    pass


class CircularDependencyError(Exception):
    """Exception levée quand une dépendance circulaire est détectée."""
    pass


class ObjectDisposedError(Exception):
    """Exception levée quand on essaie d'utiliser un objet disposé."""
    pass


# ================================
# CONTAINER GLOBAL
# ================================

# Instance globale du conteneur pour le workflow
workflow_container = EnhancedWorkflowContainer()


# ================================
# REGISTRATION HELPERS
# ================================

def register_llm_services() -> None:
    """Enregistre les services LLM dans le container de dépendances."""
    from app.services.llm.interfaces import ILLMEnricher, ITaskRouter, IConsensusValidator, ISemanticCache, EnrichmentConfig
    from app.services.llm.manager import LLMServiceManager
    from app.services.llm.task_router import IntelligentTaskRouter
    from app.services.llm.consensus_validator import ConsensusValidator
    from app.services.llm.semantic_cache import SemanticCache
    
    logger.info("Registering LLM services in DI container")
    
    # Configuration par défaut pour les services LLM
    default_config = EnrichmentConfig()
    workflow_container.register_instance(EnrichmentConfig, default_config)
    
    # Enregistrement des services LLM comme singletons pour optimiser performance
    workflow_container.register_singleton(ITaskRouter, IntelligentTaskRouter)
    workflow_container.register_singleton(IConsensusValidator, ConsensusValidator)
    workflow_container.register_singleton(ISemanticCache, SemanticCache)
    
    # Le LLMServiceManager orchestrant tout comme singleton
    workflow_container.register_singleton(ILLMEnricher, LLMServiceManager)
    
    logger.info("LLM services successfully registered", 
                services=["EnrichmentConfig", "ITaskRouter", "IConsensusValidator", "ISemanticCache", "ILLMEnricher"])


def register_all_services() -> None:
    """Enregistre tous les services du workflow."""
    logger.info("Registering all workflow services")
    
    # Services LLM
    register_llm_services()
    
    # Autres services peuvent être ajoutés ici
    # register_ocr_services()
    # register_barcode_services()
    
    logger.info("All workflow services registered successfully")