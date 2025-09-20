"""
Interfaces pour les services de lookup de code-barres.
Définit les contrats pour les services de recherche de produits alimentaires.

Architecture Pattern : Interface Segregation Principle (ISP)
Inspiration : Repository Pattern, Strategy Pattern
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class BarcodeProvider(str, Enum):
    """Providers disponibles pour lookup de codes-barres."""
    OPENFOODFACTS = "openfoodfacts"
    UPC_DATABASE = "upc_database"
    BARCODE_LOOKUP = "barcode_lookup"
    FALLBACK = "fallback"


@dataclass
class BarcodeResult:
    """
    Résultat structuré d'un lookup de code-barres.
    Contient les données produit normalisées de différentes sources.
    """
    # Identification produit
    barcode: str
    product_name: Optional[str] = None
    brand: Optional[str] = None
    categories: List[str] = None
    
    # Données nutritionnelles (pour 100g)
    nutrition_facts: Dict[str, Any] = None
    serving_size: Optional[str] = None
    
    # Informations détaillées
    ingredients: List[str] = None
    allergens: List[str] = None
    additives: List[str] = None
    labels: List[str] = None  # Bio, Fair Trade, etc.
    
    # Métadonnées
    provider: str = ""
    confidence: float = 0.0
    raw_data: Dict[str, Any] = None
    data_quality: str = "unknown"  # excellent, good, fair, poor
    
    # Informations complémentaires
    packaging: Optional[str] = None
    countries: List[str] = None
    stores: List[str] = None
    image_urls: List[str] = None
    
    def __post_init__(self):
        """Initialisation et validation des données."""
        if self.categories is None:
            self.categories = []
        if self.nutrition_facts is None:
            self.nutrition_facts = {}
        if self.ingredients is None:
            self.ingredients = []
        if self.allergens is None:
            self.allergens = []
        if self.additives is None:
            self.additives = []
        if self.labels is None:
            self.labels = []
        if self.countries is None:
            self.countries = []
        if self.stores is None:
            self.stores = []
        if self.image_urls is None:
            self.image_urls = []
        if self.raw_data is None:
            self.raw_data = {}


class IBarcodeService(ABC):
    """
    Interface principale pour les services de lookup de code-barres.
    
    Responsabilités :
    - Recherche de produits par code-barres
    - Validation de codes-barres
    - Gestion des erreurs et timeouts
    """
    
    @abstractmethod
    async def lookup_product(self, barcode: str) -> Optional[BarcodeResult]:
        """
        Recherche un produit par son code-barres.
        
        Args:
            barcode: Code-barres du produit (EAN-13, UPC-A, etc.)
            
        Returns:
            BarcodeResult avec données produit ou None si non trouvé
            
        Raises:
            BarcodeServiceError: En cas d'erreur de service
        """
        pass
    
    @abstractmethod
    async def validate_barcode(self, barcode: str) -> Dict[str, Any]:
        """
        Valide le format d'un code-barres.
        
        Args:
            barcode: Code-barres à valider
            
        Returns:
            Dict avec is_valid, format_type, warnings
        """
        pass
    
    @abstractmethod
    async def search_products(self, query: str, limit: int = 10) -> List[BarcodeResult]:
        """
        Recherche de produits par nom/marque.
        
        Args:
            query: Texte de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            Liste de BarcodeResult
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom du provider utilisé."""
        pass
    
    @property
    @abstractmethod
    def supports_nutrition_data(self) -> bool:
        """Indique si le provider fournit des données nutritionnelles."""
        pass


class IBarcodeEnricher(ABC):
    """
    Interface pour l'enrichissement des données de code-barres.
    
    Responsabilités :
    - Normalisation des données nutritionnelles
    - Enrichissement avec données externes
    - Amélioration de la qualité des données
    """
    
    @abstractmethod
    async def enrich_product_data(self, barcode_result: BarcodeResult) -> BarcodeResult:
        """
        Enrichit les données d'un produit avec informations additionnelles.
        
        Args:
            barcode_result: Résultat initial du lookup
            
        Returns:
            BarcodeResult enrichi
        """
        pass
    
    @abstractmethod
    async def normalize_nutrition_facts(self, raw_nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise les données nutritionnelles vers un format standard.
        
        Args:
            raw_nutrition: Données nutritionnelles brutes du provider
            
        Returns:
            Données nutritionnelles normalisées
        """
        pass
    
    @abstractmethod
    async def extract_allergens(self, ingredients: List[str], raw_data: Dict[str, Any]) -> List[str]:
        """
        Extrait et identifie les allergènes depuis les ingrédients.
        
        Args:
            ingredients: Liste d'ingrédients
            raw_data: Données brutes du produit
            
        Returns:
            Liste d'allergènes identifiés
        """
        pass


class BarcodeServiceError(Exception):
    """Exception pour les erreurs de service de code-barres."""
    
    def __init__(self, message: str, provider: str = "", barcode: str = "", original_error: Exception = None):
        super().__init__(message)
        self.provider = provider
        self.barcode = barcode
        self.original_error = original_error


class BarcodeValidationError(BarcodeServiceError):
    """Exception pour les erreurs de validation de code-barres."""
    pass


class BarcodeNotFoundError(BarcodeServiceError):
    """Exception quand un produit n'est pas trouvé."""
    pass


class BarcodeRateLimitError(BarcodeServiceError):
    """Exception pour les limitations de rate limit."""
    pass