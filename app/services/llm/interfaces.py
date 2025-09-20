"""
Interfaces pour le Service LLM Enrichment.

Architecture contracts suivant les patterns ISP (Interface Segregation Principle)
et Factory Pattern établis dans les services OCR/Barcode.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Providers LLM supportés avec spécialisations."""
    
    # OpenAI Models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    
    # Anthropic Models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    
    # Google Models
    GEMINI_PRO = "gemini-pro"
    GEMINI_FLASH = "gemini-1.5-flash"
    GEMINI_ULTRA = "gemini-ultra"
    
    # Local/Open Source
    LLAMA_70B = "llama-70b-instruct"
    LLAMA_8B = "llama-8b-instruct"
    
    # Fallback
    FALLBACK = "fallback"


class LLMTaskType(str, Enum):
    """Types de tâches spécialisées pour routing intelligent."""
    
    NUTRITION_ANALYSIS = "nutrition_analysis"
    ALLERGEN_DETECTION = "allergen_detection"
    HEALTH_IMPACT_ASSESSMENT = "health_impact_assessment"
    INGREDIENT_PARSING = "ingredient_parsing"
    DIETARY_COMPLIANCE = "dietary_compliance"
    PRODUCT_CATEGORIZATION = "product_categorization"
    CLAIMS_VALIDATION = "claims_validation"
    RECIPE_ANALYSIS = "recipe_analysis"
    NUTRITIONAL_COMPARISON = "nutritional_comparison"


class ConfidenceLevel(str, Enum):
    """Niveaux de confiance pour les résultats LLM."""
    
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"           # 0.8 - 0.9
    MEDIUM = "medium"       # 0.6 - 0.8
    LOW = "low"             # 0.4 - 0.6
    VERY_LOW = "very_low"   # < 0.4


@dataclass
class QualityMetrics:
    """Métriques de qualité pour évaluation LLM."""
    
    # Métriques principales
    data_completeness: float  # 0-1, toutes les données analysées?
    logical_consistency: float  # 0-1, cohérence interne?
    source_citation: float  # 0-1, sources citées correctement?
    confidence_calibration: float  # 0-1, confiance bien calibrée?
    format_compliance: float  # 0-1, format JSON respecté?
    
    # Score global
    overall_score: float = field(init=False)
    confidence_level: ConfidenceLevel = field(init=False)
    
    # Métadonnées
    analysis_duration: float = 0.0  # Secondes
    token_usage: int = 0
    provider_used: str = ""
    
    def __post_init__(self):
        """Calcul automatique du score global."""
        weights = {
            "data_completeness": 0.25,
            "logical_consistency": 0.25,
            "source_citation": 0.20,
            "confidence_calibration": 0.15,
            "format_compliance": 0.15
        }
        
        self.overall_score = (
            self.data_completeness * weights["data_completeness"] +
            self.logical_consistency * weights["logical_consistency"] +
            self.source_citation * weights["source_citation"] +
            self.confidence_calibration * weights["confidence_calibration"] +
            self.format_compliance * weights["format_compliance"]
        )
        
        # Détermination niveau de confiance
        if self.overall_score >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.overall_score >= 0.8:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.overall_score >= 0.6:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.overall_score >= 0.4:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW


@dataclass
class LLMTask:
    """Tâche LLM avec contexte et configuration."""
    
    task_type: LLMTaskType
    prompt: str
    data: Dict[str, Any]
    
    # Configuration
    preferred_provider: Optional[LLMProvider] = None
    temperature: float = 0.1  # Déterministe par défaut
    max_tokens: int = 2000
    timeout: float = 30.0
    
    # Validation
    requires_validation: bool = True
    validation_providers: List[LLMProvider] = field(default_factory=list)
    
    # Métadonnées
    priority: int = 1  # 1=high, 5=low
    retry_count: int = 0
    max_retries: int = 3
    
    def with_retry(self) -> 'LLMTask':
        """Crée une nouvelle tâche avec retry count incrémenté."""
        return LLMTask(
            task_type=self.task_type,
            prompt=self.prompt,
            data=self.data,
            preferred_provider=self.preferred_provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            requires_validation=self.requires_validation,
            validation_providers=self.validation_providers,
            priority=self.priority,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries
        )


@dataclass
class LLMResult:
    """Résultat d'analyse LLM avec métriques qualité."""
    
    # Données principales
    analysis: Dict[str, Any]
    raw_response: str
    
    # Qualité et confiance
    quality_metrics: QualityMetrics
    confidence_score: float
    
    # Métadonnées d'exécution
    provider_used: LLMProvider
    task_type: LLMTaskType
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Validation
    is_validated: bool = False
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    consensus_score: Optional[float] = None
    
    # Health Claims
    health_claims_validated: bool = False
    claims_violations: List[str] = field(default_factory=list)
    
    # Cache
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    
    @property
    def is_reliable(self) -> bool:
        """Indique si le résultat est fiable pour production."""
        return (
            self.quality_metrics.overall_score >= 0.7 and
            self.confidence_score >= 0.7 and
            (not self.requires_health_validation or self.health_claims_validated)
        )
    
    @property
    def requires_health_validation(self) -> bool:
        """Indique si validation health claims est requise."""
        return self.task_type in [
            LLMTaskType.HEALTH_IMPACT_ASSESSMENT,
            LLMTaskType.CLAIMS_VALIDATION,
            LLMTaskType.NUTRITION_ANALYSIS
        ]


@dataclass
class EnrichmentConfig:
    """Configuration pour enrichissement LLM."""
    
    # Providers à utiliser
    primary_provider: LLMProvider = LLMProvider.GPT_4O
    fallback_providers: List[LLMProvider] = field(default_factory=lambda: [
        LLMProvider.CLAUDE_3_5_SONNET,
        LLMProvider.GEMINI_PRO
    ])
    
    # Validation
    enable_consensus_validation: bool = True
    consensus_threshold: float = 0.8
    enable_health_claims_validation: bool = True
    
    # Cache
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95
    
    # Knowledge Base
    enable_knowledge_enhancement: bool = True
    knowledge_sources: List[str] = field(default_factory=lambda: [
        "anses", "usda", "efsa"
    ])
    
    # Performance
    parallel_tasks: bool = True
    max_concurrent_requests: int = 5
    timeout: float = 30.0
    
    # Quality
    min_quality_score: float = 0.7
    enable_self_correction: bool = True
    
    # Cost optimization
    enable_cost_optimization: bool = True
    max_cost_per_request: float = 0.50  # USD


# ===== CORE INTERFACES =====

class ILLMService(ABC):
    """Interface principale pour services LLM."""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom du provider LLM."""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[LLMTaskType]:
        """Types de tâches supportées."""
        pass
    
    @property
    @abstractmethod
    def cost_per_token(self) -> float:
        """Coût par token en USD."""
        pass
    
    @abstractmethod
    async def analyze(self, task: LLMTask) -> LLMResult:
        """
        Analyse une tâche LLM.
        
        Args:
            task: Tâche à analyser
            
        Returns:
            LLMResult: Résultat avec métriques qualité
        """
        pass
    
    @abstractmethod
    async def validate_response(self, response: str, expected_format: str) -> bool:
        """
        Valide la réponse LLM.
        
        Args:
            response: Réponse à valider
            expected_format: Format attendu
            
        Returns:
            bool: True si valide
        """
        pass
    
    @abstractmethod
    async def estimate_cost(self, task: LLMTask) -> float:
        """
        Estime le coût d'une tâche.
        
        Args:
            task: Tâche à estimer
            
        Returns:
            float: Coût estimé en USD
        """
        pass


class ILLMEnricher(ABC):
    """Interface pour enrichissement des données produit."""
    
    @abstractmethod
    async def enrich_product_data(self, product_data: Dict[str, Any]) -> LLMResult:
        """
        Enrichit les données d'un produit.
        
        Args:
            product_data: Données produit brutes
            
        Returns:
            LLMResult: Données enrichies
        """
        pass
    
    @abstractmethod
    async def analyze_nutrition_profile(self, nutrition_data: Dict[str, Any]) -> LLMResult:
        """
        Analyse le profil nutritionnel.
        
        Args:
            nutrition_data: Données nutritionnelles
            
        Returns:
            LLMResult: Analyse du profil
        """
        pass
    
    @abstractmethod
    async def detect_allergens_advanced(self, ingredients: List[str]) -> LLMResult:
        """
        Détection avancée d'allergènes.
        
        Args:
            ingredients: Liste d'ingrédients
            
        Returns:
            LLMResult: Allergènes détectés
        """
        pass


class ITaskRouter(ABC):
    """Interface pour routage intelligent des tâches."""
    
    @abstractmethod
    async def route_task(self, task: LLMTask) -> LLMProvider:
        """
        Route une tâche vers le provider optimal.
        
        Args:
            task: Tâche à router
            
        Returns:
            LLMProvider: Provider recommandé
        """
        pass
    
    @abstractmethod
    async def plan_analysis(self, product_data: Dict[str, Any]) -> List[LLMTask]:
        """
        Planifie l'analyse d'un produit.
        
        Args:
            product_data: Données produit
            
        Returns:
            List[LLMTask]: Tâches planifiées
        """
        pass
    
    @abstractmethod
    async def optimize_for_cost(self, tasks: List[LLMTask]) -> List[LLMTask]:
        """
        Optimise les tâches pour minimiser les coûts.
        
        Args:
            tasks: Tâches à optimiser
            
        Returns:
            List[LLMTask]: Tâches optimisées
        """
        pass


class IConsensusValidator(ABC):
    """Interface pour validation croisée entre LLMs."""
    
    @abstractmethod
    async def validate_result(self, primary_result: LLMResult, task: LLMTask) -> LLMResult:
        """
        Valide un résultat avec consensus.
        
        Args:
            primary_result: Résultat principal
            task: Tâche originale
            
        Returns:
            LLMResult: Résultat validé
        """
        pass
    
    @abstractmethod
    async def calculate_consensus_score(self, results: List[LLMResult]) -> float:
        """
        Calcule le score de consensus.
        
        Args:
            results: Résultats à comparer
            
        Returns:
            float: Score de consensus (0-1)
        """
        pass
    
    @abstractmethod
    async def resolve_conflicts(self, conflicting_results: List[LLMResult]) -> LLMResult:
        """
        Résout les conflits entre résultats.
        
        Args:
            conflicting_results: Résultats en conflit
            
        Returns:
            LLMResult: Résultat consensus
        """
        pass


class ISemanticCache(ABC):
    """Interface pour cache sémantique."""
    
    @abstractmethod
    async def get_similar_analysis(self, input_data: Dict[str, Any], threshold: float = 0.95) -> Optional[LLMResult]:
        """
        Recherche analyse similaire dans le cache.
        
        Args:
            input_data: Données d'entrée
            threshold: Seuil de similarité
            
        Returns:
            Optional[LLMResult]: Résultat similar ou None
        """
        pass
    
    @abstractmethod
    async def store_analysis(self, input_data: Dict[str, Any], result: LLMResult) -> None:
        """
        Stocke une analyse dans le cache.
        
        Args:
            input_data: Données d'entrée
            result: Résultat à stocker
        """
        pass
    
    @abstractmethod
    async def calculate_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """
        Calcule la similarité entre deux jeux de données.
        
        Args:
            data1: Premier jeu de données
            data2: Deuxième jeu de données
            
        Returns:
            float: Score de similarité (0-1)
        """
        pass


class IKnowledgeBase(ABC):
    """Interface pour base de connaissances nutritionnelles."""
    
    @abstractmethod
    async def enrich_context(self, prompt: str, data: Dict[str, Any]) -> str:
        """
        Enrichit un prompt avec connaissances contextuelles.
        
        Args:
            prompt: Prompt original
            data: Données contextuelles
            
        Returns:
            str: Prompt enrichi
        """
        pass
    
    @abstractmethod
    async def get_ingredient_info(self, ingredient: str) -> Dict[str, Any]:
        """
        Récupère informations sur un ingrédient.
        
        Args:
            ingredient: Nom de l'ingrédient
            
        Returns:
            Dict[str, Any]: Informations ingrédient
        """
        pass
    
    @abstractmethod
    async def get_allergen_patterns(self) -> Dict[str, List[str]]:
        """
        Récupère patterns d'allergènes.
        
        Returns:
            Dict[str, List[str]]: Patterns par allergène
        """
        pass


class IHealthClaimsValidator(ABC):
    """Interface pour validation des affirmations santé."""
    
    @abstractmethod
    async def validate_claims(self, analysis_result: LLMResult) -> Tuple[LLMResult, List[str]]:
        """
        Valide les affirmations santé.
        
        Args:
            analysis_result: Résultat à valider
            
        Returns:
            Tuple[LLMResult, List[str]]: Résultat sanitized + violations
        """
        pass
    
    @abstractmethod
    async def detect_violations(self, text: str) -> List[str]:
        """
        Détecte violations réglementaires.
        
        Args:
            text: Texte à analyser
            
        Returns:
            List[str]: Violations détectées
        """
        pass
    
    @abstractmethod
    async def sanitize_claims(self, text: str) -> str:
        """
        Sanitize les affirmations non conformes.
        
        Args:
            text: Texte à sanitizer
            
        Returns:
            str: Texte sanitized
        """
        pass


# ===== EXCEPTIONS =====

class LLMServiceException(Exception):
    """Exception base pour le service LLM."""
    
    def __init__(self, message: str, provider: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.timestamp = datetime.now()


class LLMValidationException(LLMServiceException):
    """Exception pour erreurs de validation."""
    
    def __init__(self, message: str, validation_errors: List[str] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or []


class LLMQuotaException(LLMServiceException):
    """Exception pour dépassement de quota."""
    
    def __init__(self, message: str, quota_type: str, current_usage: float, limit: float):
        super().__init__(message)
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.limit = limit