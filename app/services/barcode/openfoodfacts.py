"""
Implémentation du service OpenFoodFacts pour lookup de codes-barres.
Service gratuit et open source pour données produits alimentaires.

Architecture Pattern : Strategy Pattern + Async/Await
API Documentation : https://openfoodfacts.github.io/openfoodfacts-server/api/
"""

import asyncio
import aiohttp
import re
from typing import Optional, Dict, Any, List
import structlog
from urllib.parse import quote

from .interfaces import (
    IBarcodeService, BarcodeResult, BarcodeProvider,
    BarcodeServiceError, BarcodeNotFoundError, BarcodeValidationError,
    BarcodeRateLimitError
)

logger = structlog.get_logger(__name__)


class OpenFoodFactsService(IBarcodeService):
    """
    Service de lookup de produits via l'API OpenFoodFacts.
    
    Fonctionnalités :
    - Lookup par code-barres (EAN-13, UPC-A, etc.)
    - Recherche textuelle de produits
    - Données nutritionnelles normalisées
    - Support multilingue (français par défaut)
    """
    
    BASE_URL = "https://world.openfoodfacts.org/api/v2"
    SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
    
    def __init__(self, 
                 language: str = "fr",
                 country: str = "france", 
                 timeout: int = 10,
                 user_agent: str = "WellixApp/1.0"):
        """
        Initialise le service OpenFoodFacts.
        
        Args:
            language: Code langue (fr, en, es, etc.)
            country: Pays pour filtrage (france, usa, etc.)
            timeout: Timeout des requêtes en secondes
            user_agent: User-Agent pour les requêtes
        """
        self.language = language
        self.country = country
        self.timeout = timeout
        self.user_agent = user_agent
        
        # Configuration de session HTTP
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            "OpenFoodFacts service initialized",
            language=language,
            country=country,
            timeout=timeout
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtient ou crée une session HTTP."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Accept-Language": f"{self.language},en;q=0.9"
            }
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session
    
    async def close(self):
        """Ferme la session HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @property
    def provider_name(self) -> str:
        return "OpenFoodFacts"
    
    @property
    def supports_nutrition_data(self) -> bool:
        return True
    
    async def validate_barcode(self, barcode: str) -> Dict[str, Any]:
        """
        Valide le format d'un code-barres selon les standards internationaux.
        
        Supports: EAN-13, EAN-8, UPC-A, UPC-E, Code-128
        """
        if not barcode or not isinstance(barcode, str):
            return {
                "is_valid": False,
                "format_type": "unknown",
                "warnings": ["Barcode must be a non-empty string"]
            }
        
        # Nettoyage du code-barres
        clean_barcode = re.sub(r'[^0-9]', '', barcode)
        
        warnings = []
        format_type = "unknown"
        is_valid = False
        
        # Validation EAN-13 (le plus commun)
        if len(clean_barcode) == 13:
            if self._validate_ean13_checksum(clean_barcode):
                is_valid = True
                format_type = "EAN-13"
            else:
                warnings.append("Invalid EAN-13 checksum")
                format_type = "EAN-13"
        
        # Validation EAN-8
        elif len(clean_barcode) == 8:
            if self._validate_ean8_checksum(clean_barcode):
                is_valid = True
                format_type = "EAN-8"
            else:
                warnings.append("Invalid EAN-8 checksum")
                format_type = "EAN-8"
        
        # Validation UPC-A
        elif len(clean_barcode) == 12:
            if self._validate_upc_checksum(clean_barcode):
                is_valid = True
                format_type = "UPC-A"
            else:
                warnings.append("Invalid UPC-A checksum")
                format_type = "UPC-A"
        
        # Autres longueurs
        else:
            warnings.append(f"Unsupported barcode length: {len(clean_barcode)}")
        
        return {
            "is_valid": is_valid,
            "format_type": format_type,
            "cleaned_barcode": clean_barcode,
            "original_length": len(barcode),
            "cleaned_length": len(clean_barcode),
            "warnings": warnings
        }
    
    def _validate_ean13_checksum(self, barcode: str) -> bool:
        """Valide le checksum EAN-13."""
        if len(barcode) != 13:
            return False
        
        try:
            # Algorithme de validation EAN-13
            check_sum = 0
            for i, digit in enumerate(barcode[:-1]):
                weight = 1 if i % 2 == 0 else 3
                check_sum += int(digit) * weight
            
            calculated_check = (10 - (check_sum % 10)) % 10
            return calculated_check == int(barcode[-1])
        except (ValueError, IndexError):
            return False
    
    def _validate_ean8_checksum(self, barcode: str) -> bool:
        """Valide le checksum EAN-8."""
        if len(barcode) != 8:
            return False
        
        try:
            check_sum = 0
            for i, digit in enumerate(barcode[:-1]):
                weight = 3 if i % 2 == 0 else 1
                check_sum += int(digit) * weight
            
            calculated_check = (10 - (check_sum % 10)) % 10
            return calculated_check == int(barcode[-1])
        except (ValueError, IndexError):
            return False
    
    def _validate_upc_checksum(self, barcode: str) -> bool:
        """Valide le checksum UPC-A."""
        if len(barcode) != 12:
            return False
        
        try:
            check_sum = 0
            for i, digit in enumerate(barcode[:-1]):
                weight = 3 if i % 2 == 1 else 1
                check_sum += int(digit) * weight
            
            calculated_check = (10 - (check_sum % 10)) % 10
            return calculated_check == int(barcode[-1])
        except (ValueError, IndexError):
            return False
    
    async def lookup_product(self, barcode: str) -> Optional[BarcodeResult]:
        """
        Recherche un produit par code-barres via OpenFoodFacts.
        
        API Endpoint: /product/{barcode}
        """
        # Validation préalable
        validation = await self.validate_barcode(barcode)
        if not validation["is_valid"]:
            raise BarcodeValidationError(
                f"Invalid barcode format: {', '.join(validation['warnings'])}",
                provider=self.provider_name,
                barcode=barcode
            )
        
        clean_barcode = validation["cleaned_barcode"]
        
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/product/{clean_barcode}"
            
            logger.info(
                "Looking up product on OpenFoodFacts",
                barcode=clean_barcode,
                url=url
            )
            
            async with session.get(url) as response:
                if response.status == 404:
                    logger.info("Product not found", barcode=clean_barcode)
                    return None
                
                if response.status == 429:
                    raise BarcodeRateLimitError(
                        "Rate limit exceeded",
                        provider=self.provider_name,
                        barcode=clean_barcode
                    )
                
                if response.status != 200:
                    raise BarcodeServiceError(
                        f"HTTP {response.status}: {response.reason}",
                        provider=self.provider_name,
                        barcode=clean_barcode
                    )
                
                data = await response.json()
                
                # Vérification que le produit existe
                if data.get("status") != 1 or not data.get("product"):
                    logger.info("Product exists but has no data", barcode=clean_barcode)
                    return None
                
                # Conversion en BarcodeResult
                return await self._parse_openfoodfacts_response(clean_barcode, data["product"])
        
        except aiohttp.ClientError as e:
            raise BarcodeServiceError(
                f"Network error: {str(e)}",
                provider=self.provider_name,
                barcode=clean_barcode,
                original_error=e
            )
        except asyncio.TimeoutError:
            raise BarcodeServiceError(
                "Request timeout",
                provider=self.provider_name,
                barcode=clean_barcode
            )
    
    async def _parse_openfoodfacts_response(self, barcode: str, product_data: Dict[str, Any]) -> BarcodeResult:
        """Parse la réponse OpenFoodFacts en BarcodeResult."""
        
        # Extraction des informations de base
        product_name = (
            product_data.get("product_name_fr") or 
            product_data.get("product_name") or 
            product_data.get("generic_name") or
            "Unknown Product"
        )
        
        brand = product_data.get("brands", "").split(",")[0].strip() if product_data.get("brands") else None
        
        # Catégories
        categories = []
        if product_data.get("categories"):
            categories = [cat.strip() for cat in product_data["categories"].split(",")]
        
        # Données nutritionnelles (pour 100g)
        nutrition_facts = {}
        nutriments = product_data.get("nutriments", {})
        
        # Mapping des nutriments OpenFoodFacts vers format standard
        nutrient_mapping = {
            "energy-kcal_100g": "calories",
            "proteins_100g": "protein", 
            "carbohydrates_100g": "carbohydrates",
            "fat_100g": "total_fat",
            "saturated-fat_100g": "saturated_fat",
            "trans-fat_100g": "trans_fat",
            "fiber_100g": "fiber",
            "sugars_100g": "sugar",
            "salt_100g": "sodium",  # Sera converti de sel vers sodium
            "sodium_100g": "sodium"
        }
        
        for off_key, standard_key in nutrient_mapping.items():
            if off_key in nutriments and nutriments[off_key] is not None:
                value = nutriments[off_key]
                # Conversion sel -> sodium (1g sel = 0.4g sodium)
                if off_key == "salt_100g" and "sodium_100g" not in nutriments:
                    value = value * 0.4
                nutrition_facts[standard_key] = float(value)
        
        # Ingrédients
        ingredients = []
        if product_data.get("ingredients_text_fr"):
            ingredients_text = product_data["ingredients_text_fr"]
        elif product_data.get("ingredients_text"):
            ingredients_text = product_data["ingredients_text"]
        else:
            ingredients_text = ""
        
        if ingredients_text:
            # Parse simple des ingrédients (séparation par virgules)
            ingredients = [ing.strip() for ing in ingredients_text.split(",") if ing.strip()]
        
        # Allergènes
        allergens = []
        if product_data.get("allergens"):
            allergens_text = product_data["allergens"]
            # Les allergènes OpenFoodFacts sont préfixés par "en:"
            allergens = [
                allergen.replace("en:", "").strip() 
                for allergen in allergens_text.split(",") 
                if allergen.strip()
            ]
        
        # Additifs
        additives = []
        if product_data.get("additives_tags"):
            additives = [
                additive.replace("en:", "").strip()
                for additive in product_data["additives_tags"]
            ]
        
        # Labels (Bio, Fair Trade, etc.)
        labels = []
        if product_data.get("labels"):
            labels = [label.strip() for label in product_data["labels"].split(",")]
        
        # URLs d'images
        image_urls = []
        if product_data.get("image_url"):
            image_urls.append(product_data["image_url"])
        if product_data.get("image_front_url"):
            image_urls.append(product_data["image_front_url"])
        if product_data.get("image_nutrition_url"):
            image_urls.append(product_data["image_nutrition_url"])
        
        # Évaluation de la qualité des données
        data_quality = self._assess_data_quality(product_data, nutrition_facts, ingredients)
        
        # Calcul de la confiance
        confidence = self._calculate_confidence(product_data, nutrition_facts, ingredients)
        
        return BarcodeResult(
            barcode=barcode,
            product_name=product_name,
            brand=brand,
            categories=categories,
            nutrition_facts=nutrition_facts,
            serving_size=product_data.get("serving_size"),
            ingredients=ingredients,
            allergens=allergens,
            additives=additives,
            labels=labels,
            provider=self.provider_name,
            confidence=confidence,
            raw_data=product_data,
            data_quality=data_quality,
            packaging=product_data.get("packaging"),
            countries=product_data.get("countries", "").split(",") if product_data.get("countries") else [],
            stores=product_data.get("stores", "").split(",") if product_data.get("stores") else [],
            image_urls=image_urls
        )
    
    def _assess_data_quality(self, product_data: Dict[str, Any], nutrition_facts: Dict[str, Any], ingredients: List[str]) -> str:
        """Évalue la qualité des données du produit."""
        score = 0
        max_score = 10
        
        # Nom du produit (2 points)
        if product_data.get("product_name_fr") or product_data.get("product_name"):
            score += 2
        
        # Marque (1 point)
        if product_data.get("brands"):
            score += 1
        
        # Données nutritionnelles (3 points)
        essential_nutrients = ["calories", "protein", "carbohydrates", "total_fat"]
        nutrition_score = sum(1 for nutrient in essential_nutrients if nutrient in nutrition_facts)
        score += min(3, nutrition_score)
        
        # Ingrédients (2 points)
        if len(ingredients) >= 3:
            score += 2
        elif len(ingredients) >= 1:
            score += 1
        
        # Images (1 point)
        if product_data.get("image_url") or product_data.get("image_front_url"):
            score += 1
        
        # Catégories (1 point)
        if product_data.get("categories"):
            score += 1
        
        # Conversion en qualité textuelle
        ratio = score / max_score
        if ratio >= 0.8:
            return "excellent"
        elif ratio >= 0.6:
            return "good"
        elif ratio >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_confidence(self, product_data: Dict[str, Any], nutrition_facts: Dict[str, Any], ingredients: List[str]) -> float:
        """Calcule un score de confiance basé sur la complétude des données."""
        confidence_factors = {
            "has_product_name": 0.2,
            "has_brand": 0.1,
            "has_nutrition": 0.3,
            "has_ingredients": 0.2,
            "has_image": 0.1,
            "data_freshness": 0.1
        }
        
        confidence = 0.0
        
        # Nom du produit
        if product_data.get("product_name_fr") or product_data.get("product_name"):
            confidence += confidence_factors["has_product_name"]
        
        # Marque
        if product_data.get("brands"):
            confidence += confidence_factors["has_brand"]
        
        # Données nutritionnelles
        if nutrition_facts:
            nutrition_completeness = len(nutrition_facts) / 8  # 8 nutriments principaux
            confidence += confidence_factors["has_nutrition"] * min(1.0, nutrition_completeness)
        
        # Ingrédients
        if ingredients:
            ingredient_completeness = min(1.0, len(ingredients) / 5)  # 5+ ingrédients = complet
            confidence += confidence_factors["has_ingredients"] * ingredient_completeness
        
        # Images
        if product_data.get("image_url"):
            confidence += confidence_factors["has_image"]
        
        # Fraîcheur des données (basée sur last_modified_t)
        if product_data.get("last_modified_t"):
            import time
            last_modified = product_data["last_modified_t"]
            current_time = time.time()
            days_old = (current_time - last_modified) / (24 * 3600)
            
            # Données récentes = plus de confiance
            if days_old < 30:
                freshness_factor = 1.0
            elif days_old < 180:
                freshness_factor = 0.8
            elif days_old < 365:
                freshness_factor = 0.6
            else:
                freshness_factor = 0.3
            
            confidence += confidence_factors["data_freshness"] * freshness_factor
        
        return min(1.0, confidence)
    
    async def search_products(self, query: str, limit: int = 10) -> List[BarcodeResult]:
        """
        Recherche de produits par nom/marque via OpenFoodFacts.
        
        API Endpoint: /cgi/search.pl
        """
        if not query or not query.strip():
            return []
        
        try:
            session = await self._get_session()
            
            params = {
                "search_terms": query.strip(),
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": min(limit, 50),  # OpenFoodFacts limite à 50
                "fields": "code,product_name,brands,categories,image_url,nutriments"
            }
            
            logger.info(
                "Searching products on OpenFoodFacts",
                query=query,
                limit=limit
            )
            
            async with session.get(self.SEARCH_URL, params=params) as response:
                if response.status != 200:
                    raise BarcodeServiceError(
                        f"Search failed: HTTP {response.status}",
                        provider=self.provider_name
                    )
                
                data = await response.json()
                products = data.get("products", [])
                
                # Conversion en BarcodeResult
                results = []
                for product in products[:limit]:
                    if product.get("code"):  # Assurer qu'il y a un code-barres
                        try:
                            result = await self._parse_openfoodfacts_response(
                                product["code"], 
                                product
                            )
                            results.append(result)
                        except Exception as e:
                            logger.warning(
                                "Failed to parse search result", 
                                product_code=product.get("code"),
                                error=str(e)
                            )
                
                logger.info(
                    "Product search completed",
                    query=query,
                    found_products=len(results)
                )
                
                return results
        
        except aiohttp.ClientError as e:
            raise BarcodeServiceError(
                f"Search network error: {str(e)}",
                provider=self.provider_name,
                original_error=e
            )
        except asyncio.TimeoutError:
            raise BarcodeServiceError(
                "Search request timeout",
                provider=self.provider_name
            )