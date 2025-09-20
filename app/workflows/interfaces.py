"""
Workflow interfaces selon les principes SOLID.
Définit les contrats pour l'architecture workflow découplee et testable.

Architecture Pattern : Command + Strategy + Observer + Chain of Responsibility
Inspiration : Temporal.io, Uber Cadence, Meta Workflow Engine
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypedDict, Union, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from app.models.health import NutritionData, UserHealthContext


class WorkflowStage(Enum):
    """Étapes du workflow d'analyse alimentaire - State Machine Pattern."""
    ENTRY_ROUTING = "entry_routing"
    INPUT_VALIDATION = "input_validation"
    TOKEN_VALIDATION = "token_validation"
    DATA_EXTRACTION = "data_extraction"
    NUTRITION_ENRICHMENT = "nutrition_enrichment"
    HEALTH_PROFILE_CONTEXT = "health_profile_context"
    EXPERT_ANALYSIS = "expert_analysis"
    SCORE_CALCULATION = "score_calculation"
    ALTERNATIVE_GENERATION = "alternative_generation"
    CHAT_CONTEXT_PREPARATION = "chat_context_preparation"
    RESPONSE_ASSEMBLY = "response_assembly"


class InputType(Enum):
    """Types d'input supportés."""
    IMAGE = "image"
    BARCODE = "barcode" 
    JSON_DATA = "json_data"
    MIXED = "mixed"


class QualityLevel(Enum):
    """Niveaux de qualité d'analyse."""
    BASIC = "basic"           # Algorithmic only
    STANDARD = "standard"     # Light LLM enhancement
    PREMIUM = "premium"       # Full LLM + context
    EXPERT = "expert"         # Maximum intelligence


class AlternativeType(Enum):
    """Types d'alternatives à générer."""
    HEALTHIER = "healthier"   # Meilleure santé
    CHEAPER = "cheaper"       # Moins cher
    NEARBY = "nearby"         # Proximité géographique


@dataclass(frozen=True)  # Immutable pour thread safety
class ProcessingStep:
    """Étape de traitement dans l'historique."""
    stage: str
    timestamp: datetime
    duration_ms: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowError:
    """Erreur survenue dans le workflow."""
    stage: str
    error_type: str
    message: str
    timestamp: datetime
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Métriques de performance du workflow."""
    total_duration_ms: float
    stage_durations: Dict[str, float]
    memory_peak_mb: Optional[float] = None
    api_calls_count: int = 0
    cache_hit_rate: float = 0.0


@dataclass(frozen=True)
class InputData:
    """Données d'entrée du workflow - immutable."""
    type: InputType
    image_data: Optional[bytes] = None
    barcode: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    complexity_hints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation des données d'entrée."""
        if self.type == InputType.IMAGE and not self.image_data:
            raise ValueError("Image data required for IMAGE input type")
        if self.type == InputType.BARCODE and not self.barcode:
            raise ValueError("Barcode required for BARCODE input type")
        if self.type == InputType.JSON_DATA and not self.json_data:
            raise ValueError("JSON data required for JSON_DATA input type")


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration de l'analyse - Strategy Pattern."""
    quality_level: QualityLevel
    enable_alternatives: bool = True
    max_alternatives_per_type: int = 3
    enable_chat_context: bool = True
    custom_preferences: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def requires_llm(self) -> bool:
        """Détermine si cette configuration nécessite le LLM."""
        return self.quality_level in [QualityLevel.PREMIUM, QualityLevel.EXPERT]


@dataclass(frozen=True)
class WorkflowState:
    """État immutable du workflow - Command Pattern State."""
    # === CORE DATA (Required) ===
    workflow_id: str
    user_context: UserHealthContext
    input_data: InputData
    analysis_config: AnalysisConfig
    
    # === PROCESSING STATE (Optional - build progressively) ===
    current_stage: WorkflowStage = WorkflowStage.ENTRY_ROUTING
    nutrition_data: Optional[NutritionData] = None
    expert_analysis: Optional[Dict[str, Any]] = None
    calculated_score: Optional[int] = None
    alternatives: Optional[Dict[AlternativeType, List[Dict[str, Any]]]] = None
    chat_context: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
    
    # === METADATA (Immutable tracking) ===
    processing_history: tuple = field(default_factory=tuple)
    error_history: tuple = field(default_factory=tuple)
    performance_metrics: Optional[PerformanceMetrics] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # === STATE TRANSITION METHODS (Return new instances) ===
    def advance_to_stage(self, stage: WorkflowStage) -> 'WorkflowState':
        """Advance workflow to next stage with timing."""
        from dataclasses import replace
        step = ProcessingStep(
            stage=stage.value,
            timestamp=datetime.utcnow()
        )
        return replace(
            self,
            current_stage=stage,
            processing_history=self.processing_history + (step,)
        )
    
    def with_nutrition_data(self, data: NutritionData) -> 'WorkflowState':
        """Add nutrition data to state."""
        from dataclasses import replace
        return replace(self, nutrition_data=data)
    
    def with_expert_analysis(self, analysis: Dict[str, Any]) -> 'WorkflowState':
        """Add expert analysis to state."""
        from dataclasses import replace
        return replace(self, expert_analysis=analysis)
    
    def with_score(self, score: int) -> 'WorkflowState':
        """Add calculated score to state."""
        from dataclasses import replace
        return replace(self, calculated_score=score)
    
    def with_alternatives(self, alternatives: Dict[AlternativeType, List[Dict[str, Any]]]) -> 'WorkflowState':
        """Add alternatives to state."""
        from dataclasses import replace
        return replace(self, alternatives=alternatives)
    
    def with_chat_context(self, context: Dict[str, Any]) -> 'WorkflowState':
        """Add chat context to state."""
        from dataclasses import replace
        return replace(self, chat_context=context)
    
    def with_error(self, error: WorkflowError) -> 'WorkflowState':
        """Add error to state history."""
        from dataclasses import replace
        return replace(
            self,
            error_history=self.error_history + (error,)
        )
    
    def with_final_response(self, response: Dict[str, Any]) -> 'WorkflowState':
        """Set final response."""
        from dataclasses import replace
        return replace(self, final_response=response)
    
    # === UTILITY PROPERTIES ===
    @property
    def has_errors(self) -> bool:
        """Check if workflow has any errors."""
        return len(self.error_history) > 0
    
    @property
    def last_error(self) -> Optional[WorkflowError]:
        """Get the most recent error."""
        return self.error_history[-1] if self.error_history else None
    
    @property
    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        return self.final_response is not None
    
    @property
    def processing_time_ms(self) -> float:
        """Calculate total processing time so far."""
        if not self.processing_history:
            return 0.0
        
        start_time = self.created_at
        last_step = self.processing_history[-1]
        return (last_step.timestamp - start_time).total_seconds() * 1000


@dataclass
class WorkflowResult:
    """Résultat final du workflow - immutable output."""
    success: bool
    workflow_id: str
    data: Optional[Dict[str, Any]]
    errors: List[WorkflowError]
    performance_metrics: PerformanceMetrics
    processing_history: List[ProcessingStep]
    
    @classmethod
    def success_result(cls, 
                      workflow_id: str, 
                      data: Dict[str, Any], 
                      metrics: PerformanceMetrics,
                      history: List[ProcessingStep]) -> 'WorkflowResult':
        """Create successful workflow result."""
        return cls(
            success=True,
            workflow_id=workflow_id,
            data=data,
            errors=[],
            performance_metrics=metrics,
            processing_history=history
        )
    
    @classmethod
    def error_result(cls,
                    workflow_id: str,
                    errors: List[WorkflowError],
                    partial_data: Optional[Dict[str, Any]] = None) -> 'WorkflowResult':
        """Create error workflow result."""
        return cls(
            success=False,
            workflow_id=workflow_id,
            data=partial_data,
            errors=errors,
            performance_metrics=PerformanceMetrics(total_duration_ms=0.0, stage_durations={}),
            processing_history=[]
        )


# ================================
# CORE WORKFLOW INTERFACES
# ================================

class IWorkflowNode(ABC):
    """
    Interface pour un nœud de workflow - Single Responsibility Principle.
    
    Chaque node a UNE responsabilité et peut être testé indépendamment.
    Pattern : Command + Strategy
    """
    
    @property
    @abstractmethod
    def node_name(self) -> str:
        """Nom du nœud pour le logging/debugging."""
        pass
    
    @property
    @abstractmethod
    def required_stage(self) -> WorkflowStage:
        """Stage requis pour exécuter ce node."""
        pass
    
    @abstractmethod
    async def can_process(self, state: WorkflowState) -> bool:
        """
        Vérifie si ce nœud peut traiter l'état actuel.
        
        Args:
            state: État du workflow
            
        Returns:
            True si le nœud peut traiter cet état
        """
        pass
    
    @abstractmethod
    async def process(self, state: WorkflowState) -> WorkflowState:
        """
        Traite l'état et retourne l'état modifié.
        
        Args:
            state: État actuel du workflow
            
        Returns:
            État modifié après traitement
            
        Raises:
            WorkflowNodeError: Si le traitement échoue
        """
        pass
    
    @abstractmethod
    async def validate_preconditions(self, state: WorkflowState) -> List[str]:
        """
        Valide les préconditions avant traitement.
        
        Returns:
            Liste des erreurs de validation (vide si OK)
        """
        pass


class IWorkflowOrchestrator(ABC):
    """
    Interface pour l'orchestrateur de workflow - Open/Closed Principle.
    
    Coordonne l'exécution des nodes selon la state machine.
    Pattern : Template Method + Chain of Responsibility
    """
    
    @abstractmethod
    async def execute(
        self,
        input_data: InputData,
        user_context: UserHealthContext,
        analysis_config: AnalysisConfig
    ) -> WorkflowResult:
        """
        Exécute le workflow complet.
        
        Args:
            input_data: Données d'entrée (image, barcode, JSON)
            user_context: Contexte santé de l'utilisateur
            analysis_config: Configuration de l'analyse
            
        Returns:
            Résultat final du workflow
        """
        pass
    
    @abstractmethod
    def register_node(self, node: IWorkflowNode) -> None:
        """
        Enregistre un nœud dans l'orchestrateur.
        
        Args:
            node: Nœud à enregistrer
        """
        pass
    
    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Récupère le statut actuel d'un workflow.
        
        Args:
            workflow_id: ID du workflow
            
        Returns:
            État actuel ou None si non trouvé
        """
        pass


class IWorkflowRouter(ABC):
    """
    Interface pour le routage du workflow - Interface Segregation.
    
    Détermine les transitions entre stages selon la logique business.
    Pattern : State Machine + Strategy
    """
    
    @abstractmethod
    def get_next_stage(self, current_state: WorkflowState) -> Optional[WorkflowStage]:
        """
        Détermine la prochaine étape basée sur l'état actuel.
        
        Args:
            current_state: État actuel du workflow
            
        Returns:
            Prochaine étape ou None si terminé
        """
        pass
    
    @abstractmethod
    def should_skip_stage(self, stage: WorkflowStage, state: WorkflowState) -> bool:
        """
        Détermine si une étape doit être ignorée.
        
        Args:
            stage: Étape à évaluer
            state: État actuel
            
        Returns:
            True si l'étape doit être ignorée
        """
        pass
    
    @abstractmethod
    def can_transition_to(self, from_stage: WorkflowStage, to_stage: WorkflowStage) -> bool:
        """
        Vérifie si une transition est valide.
        
        Args:
            from_stage: Stage actuel
            to_stage: Stage de destination
            
        Returns:
            True si la transition est autorisée
        """
        pass


class IWorkflowValidator(ABC):
    """
    Interface pour la validation des données - Single Responsibility.
    
    Valide les données et transitions du workflow.
    Pattern : Chain of Responsibility + Specification
    """
    
    @abstractmethod
    async def validate_input(self, input_data: InputData) -> List[str]:
        """
        Valide les données d'entrée.
        
        Returns:
            Liste des erreurs de validation (vide si valide)
        """
        pass
    
    @abstractmethod
    async def validate_state_transition(
        self,
        from_stage: WorkflowStage,
        to_stage: WorkflowStage,
        state: WorkflowState
    ) -> List[str]:
        """
        Valide qu'une transition d'état est valide.
        
        Returns:
            Liste des erreurs de validation (vide si valide)
        """
        pass
    
    @abstractmethod
    async def validate_user_context(self, user_context: UserHealthContext) -> List[str]:
        """
        Valide le contexte utilisateur.
        
        Returns:
            Liste des erreurs de validation (vide si valide)
        """
        pass


# ================================
# BUSINESS LOGIC INTERFACES
# ================================

class IExtractionStrategy(ABC):
    """Interface pour les stratégies d'extraction - Strategy Pattern."""
    
    @abstractmethod
    async def extract(self, input_data: InputData) -> Dict[str, Any]:
        """Extract nutrition data from input."""
        pass
    
    @property
    @abstractmethod
    def supported_input_types(self) -> List[InputType]:
        """Types d'input supportés par cette stratégie."""
        pass


class IAnalysisStrategy(ABC):
    """Interface pour les stratégies d'analyse - Strategy Pattern."""
    
    @abstractmethod
    async def analyze(self, 
                     nutrition_data: NutritionData, 
                     user_context: UserHealthContext,
                     quality_level: QualityLevel) -> Dict[str, Any]:
        """Perform health analysis."""
        pass
    
    @property
    @abstractmethod
    def supported_quality_levels(self) -> List[QualityLevel]:
        """Niveaux de qualité supportés."""
        pass


class IAlternativeStrategy(ABC):
    """Interface pour les stratégies d'alternatives - Strategy Pattern."""
    
    @abstractmethod
    async def generate_alternatives(self,
                                   reference_nutrition: NutritionData,
                                   user_context: UserHealthContext,
                                   max_count: int = 3) -> List[Dict[str, Any]]:
        """Generate product alternatives."""
        pass
    
    @property
    @abstractmethod
    def alternative_type(self) -> AlternativeType:
        """Type d'alternative généré par cette stratégie."""
        pass


# ================================
# EXTERNAL SERVICE INTERFACES  
# ================================

class ITokenService(ABC):
    """Interface pour le service de tokens - Dependency Inversion."""
    
    @abstractmethod
    async def validate_and_reserve(self, 
                                  user_id: str, 
                                  cost: int, 
                                  feature_type: str) -> Dict[str, Any]:
        """Validate and reserve tokens for operation."""
        pass
    
    @abstractmethod
    async def consume_tokens(self, user_id: str, cost: int, transaction_id: str) -> bool:
        """Consume reserved tokens."""
        pass
    
    @abstractmethod
    async def release_tokens(self, user_id: str, transaction_id: str) -> bool:
        """Release reserved tokens (on error)."""
        pass


class ILLMOrchestrator(ABC):
    """Interface pour l'orchestration LLM - Dependency Inversion."""
    
    @abstractmethod
    async def analyze_nutrition(self,
                              nutrition_data: NutritionData,
                              user_context: UserHealthContext,
                              analysis_prompt: str) -> Dict[str, Any]:
        """Perform LLM-enhanced nutrition analysis."""
        pass
    
    @abstractmethod
    async def generate_alternatives_suggestions(self,
                                              current_product: NutritionData,
                                              user_preferences: Dict[str, Any]) -> List[str]:
        """Generate alternative product suggestions."""
        pass


class ICacheManager(ABC):
    """Interface pour la gestion du cache - Dependency Inversion."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass


class IEventBus(ABC):
    """Interface pour l'event bus - Observer Pattern."""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event to bus."""
        pass
    
    @abstractmethod
    def subscribe(self, 
                 event_type: str, 
                 handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to event type."""
        pass


class IMetricsCollector(ABC):
    """Interface pour la collecte de métriques - Observer Pattern."""
    
    @abstractmethod
    def increment(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        pass
    
    @abstractmethod
    def histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        pass
    
    @abstractmethod
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric."""
        pass


# ================================
# EXCEPTIONS
# ================================

class WorkflowNodeError(Exception):
    """Exception spécifique aux nœuds de workflow."""
    
    def __init__(self, node_name: str, message: str, original_error: Optional[Exception] = None, recoverable: bool = True):
        self.node_name = node_name
        self.original_error = original_error
        self.recoverable = recoverable
        super().__init__(f"Node '{node_name}': {message}")


class WorkflowValidationError(Exception):
    """Exception pour les erreurs de validation du workflow."""
    
    def __init__(self, errors: List[str], stage: Optional[str] = None):
        self.errors = errors
        self.stage = stage
        super().__init__(f"Workflow validation failed: {', '.join(errors)}")


class WorkflowStateError(Exception):
    """Exception pour les erreurs d'état du workflow."""
    
    def __init__(self, message: str, current_stage: Optional[WorkflowStage] = None):
        self.current_stage = current_stage
        super().__init__(message)


class WorkflowTimeoutError(Exception):
    """Exception pour les timeouts du workflow."""
    
    def __init__(self, stage: str, timeout_seconds: float):
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Workflow stage '{stage}' timed out after {timeout_seconds}s")