"""
OpenFoodFacts API integration for product data enrichment.
"""

import httpx
from typing import Dict, Any, Optional, List
import structlog

from app.core.config import settings
from app.cache.cache_manager import cache_manager

logger = structlog.get_logger(__name__)


class OpenFoodFactsService:
    """OpenFoodFacts API service for product data enrichment."""
    
    def __init__(self):
        self.base_url = "https://world.openfoodfacts.org/api/v2"
        self.user_agent = settings.openfoodfacts_user_agent
        self.timeout = 10.0
    
    async def get_product_by_barcode(
        self,
        barcode: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get product data from OpenFoodFacts by barcode.
        
        Args:
            barcode: Product barcode (EAN-13, UPC, etc.)
            use_cache: Whether to use cached results
            
        Returns:
            Product data dictionary or None if not found
        """
        # Check cache first
        if use_cache:
            cached_product = await cache_manager.get_openfoodfacts_product(barcode)
            if cached_product:
                logger.info(f"OpenFoodFacts product {barcode} retrieved from cache")
                return cached_product
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/product/{barcode}"
                headers = {"User-Agent": self.user_agent}
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") != 1:
                    logger.info(f"Product {barcode} not found in OpenFoodFacts")
                    return None
                
                product_data = self._process_product_data(data.get("product", {}))
                
                # Cache the result
                if use_cache and product_data:
                    await cache_manager.set_openfoodfacts_product(barcode, product_data)
                
                logger.info(f"Retrieved product data for barcode {barcode}")
                return product_data
                
        except httpx.TimeoutException:
            logger.error(f"Timeout retrieving product {barcode} from OpenFoodFacts")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error retrieving product {barcode}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving product {barcode} from OpenFoodFacts: {e}")
            return None
    
    async def search_products(
        self,
        query: str,
        limit: int = 20,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products in OpenFoodFacts.
        
        Args:
            query: Search query
            limit: Maximum number of results
            categories: Optional category filters
            
        Returns:
            List of product data dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/search"
                headers = {"User-Agent": self.user_agent}
                
                params = {
                    "search_terms": query,
                    "page_size": min(limit, 100),
                    "json": 1
                }
                
                if categories:
                    params["categories"] = ",".join(categories)
                
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                products = data.get("products", [])
                
                # Process each product
                processed_products = []
                for product in products:
                    processed_product = self._process_product_data(product)
                    if processed_product:
                        processed_products.append(processed_product)
                
                logger.info(f"Found {len(processed_products)} products for query: {query}")
                return processed_products
                
        except Exception as e:
            logger.error(f"Error searching products in OpenFoodFacts: {e}")
            return []
    
    def _process_product_data(self, raw_product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process raw OpenFoodFacts product data into standardized format.
        
        Args:
            raw_product: Raw product data from OpenFoodFacts API
            
        Returns:
            Processed product data dictionary
        """
        if not raw_product:
            return None
        
        try:
            # Basic product information
            product = {
                "barcode": raw_product.get("code"),
                "product_name": raw_product.get("product_name", "").strip(),
                "brand": raw_product.get("brands", "").strip(),
                "categories": self._extract_categories(raw_product.get("categories", "")),
                "labels": self._extract_labels(raw_product.get("labels", "")),
                
                # Nutrition data
                "nutrition_data": self._extract_nutrition_data(raw_product),
                
                # Ingredients and allergens
                "ingredients": self._extract_ingredients(raw_product),
                "allergens": self._extract_allergens(raw_product),
                "additives": self._extract_additives(raw_product),
                
                # Quality scores
                "nutriscore": raw_product.get("nutriscore_grade", "").upper(),
                "nova_group": raw_product.get("nova_group"),
                "ecoscore": raw_product.get("ecoscore_grade", "").upper(),
                
                # Additional data
                "serving_size": raw_product.get("serving_size"),
                "packaging": raw_product.get("packaging", "").strip(),
                "countries": raw_product.get("countries", "").strip(),
                
                # Images
                "image_url": raw_product.get("image_url"),
                "image_nutrition_url": raw_product.get("image_nutrition_url"),
                
                # Metadata
                "data_quality": self._assess_data_quality(raw_product),
                "last_modified": raw_product.get("last_modified_t"),
                "completeness": raw_product.get("completeness", 0)
            }
            
            # Only return if we have essential data
            if product["product_name"] or product["barcode"]:
                return product
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing OpenFoodFacts product data: {e}")
            return None
    
    def _extract_nutrition_data(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize nutrition data."""
        nutriments = raw_product.get("nutriments", {})
        
        nutrition = {}
        
        # Mapping of OpenFoodFacts keys to our standard keys
        nutrient_mapping = {
            "energy-kcal_100g": "calories_per_100g",
            "energy-kcal_serving": "calories_per_serving",
            "proteins_100g": "protein_per_100g",
            "proteins_serving": "protein_per_serving",
            "carbohydrates_100g": "carbohydrates_per_100g",
            "carbohydrates_serving": "carbohydrates_per_serving",
            "fat_100g": "total_fat_per_100g",
            "fat_serving": "total_fat_per_serving",
            "saturated-fat_100g": "saturated_fat_per_100g",
            "saturated-fat_serving": "saturated_fat_per_serving",
            "trans-fat_100g": "trans_fat_per_100g",
            "trans-fat_serving": "trans_fat_per_serving",
            "fiber_100g": "fiber_per_100g",
            "fiber_serving": "fiber_per_serving",
            "sugars_100g": "sugar_per_100g",
            "sugars_serving": "sugar_per_serving",
            "salt_100g": "salt_per_100g",
            "salt_serving": "salt_per_serving",
            "sodium_100g": "sodium_per_100g",
            "sodium_serving": "sodium_per_serving",
            "potassium_100g": "potassium_per_100g",
            "potassium_serving": "potassium_per_serving",
            "cholesterol_100g": "cholesterol_per_100g",
            "cholesterol_serving": "cholesterol_per_serving"
        }
        
        for off_key, our_key in nutrient_mapping.items():
            value = nutriments.get(off_key)
            if value is not None:
                try:
                    nutrition[our_key] = float(value)
                except (ValueError, TypeError):
                    continue
        
        # Convert salt to sodium if sodium not available
        if "sodium_per_100g" not in nutrition and "salt_per_100g" in nutrition:
            # Salt to sodium conversion: sodium = salt / 2.5
            nutrition["sodium_per_100g"] = nutrition["salt_per_100g"] / 2.5
        
        return nutrition
    
    def _extract_categories(self, categories_str: str) -> List[str]:
        """Extract and clean categories."""
        if not categories_str:
            return []
        
        categories = [cat.strip() for cat in categories_str.split(",")]
        # Remove language prefixes and clean
        cleaned_categories = []
        for cat in categories:
            if ":" in cat:
                cat = cat.split(":")[-1]
            cleaned_categories.append(cat.strip())
        
        return cleaned_categories[:10]  # Limit to 10 categories
    
    def _extract_labels(self, labels_str: str) -> List[str]:
        """Extract and clean labels."""
        if not labels_str:
            return []
        
        labels = [label.strip() for label in labels_str.split(",")]
        # Remove language prefixes and clean
        cleaned_labels = []
        for label in labels:
            if ":" in label:
                label = label.split(":")[-1]
            cleaned_labels.append(label.strip())
        
        return cleaned_labels[:15]  # Limit to 15 labels
    
    def _extract_ingredients(self, raw_product: Dict[str, Any]) -> List[str]:
        """Extract ingredients list."""
        ingredients_text = raw_product.get("ingredients_text", "")
        if not ingredients_text:
            return []
        
        # Simple ingredient extraction - split by common separators
        import re
        ingredients = re.split(r'[,;]', ingredients_text)
        cleaned_ingredients = []
        
        for ingredient in ingredients:
            # Clean up ingredient text
            ingredient = re.sub(r'\([^)]*\)', '', ingredient)  # Remove parentheses
            ingredient = re.sub(r'\d+%?', '', ingredient)  # Remove percentages
            ingredient = ingredient.strip()
            
            if ingredient and len(ingredient) > 2:
                cleaned_ingredients.append(ingredient)
        
        return cleaned_ingredients[:30]  # Limit to 30 ingredients
    
    def _extract_allergens(self, raw_product: Dict[str, Any]) -> List[str]:
        """Extract allergens list."""
        allergens_str = raw_product.get("allergens", "")
        if not allergens_str:
            return []
        
        allergens = [allergen.strip() for allergen in allergens_str.split(",")]
        # Remove language prefixes
        cleaned_allergens = []
        for allergen in allergens:
            if ":" in allergen:
                allergen = allergen.split(":")[-1]
            cleaned_allergens.append(allergen.strip())
        
        return cleaned_allergens
    
    def _extract_additives(self, raw_product: Dict[str, Any]) -> List[str]:
        """Extract additives list."""
        additives = raw_product.get("additives_tags", [])
        if not additives:
            return []
        
        # Clean additive tags
        cleaned_additives = []
        for additive in additives:
            if additive.startswith("en:"):
                additive = additive[3:]  # Remove "en:" prefix
            cleaned_additives.append(additive.replace("-", " ").title())
        
        return cleaned_additives[:20]  # Limit to 20 additives
    
    def _assess_data_quality(self, raw_product: Dict[str, Any]) -> str:
        """Assess the quality of product data."""
        score = 0
        
        # Check for essential fields
        if raw_product.get("product_name"):
            score += 20
        if raw_product.get("brands"):
            score += 15
        if raw_product.get("ingredients_text"):
            score += 20
        if raw_product.get("nutriments", {}).get("energy-kcal_100g"):
            score += 25
        if raw_product.get("image_url"):
            score += 10
        if raw_product.get("categories"):
            score += 10
        
        if score >= 80:
            return "high"
        elif score >= 50:
            return "medium"
        else:
            return "low"


# Global OpenFoodFacts service instance
openfoodfacts_service = OpenFoodFactsService()
