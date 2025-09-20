"""
Enrichisseur de données pour les résultats de lookup de codes-barres.
Normalise et améliore la qualité des données produits.

Architecture Pattern : Decorator Pattern + Strategy Pattern
Inspiration : Data Transformation Pipeline, Enrichment Pattern
"""

import re
import asyncio
from typing import Dict, Any, List, Optional, Set
import structlog

from .interfaces import IBarcodeEnricher, BarcodeResult

logger = structlog.get_logger(__name__)


class NutritionDataEnricher(IBarcodeEnricher):
    """
    Enrichisseur spécialisé pour les données nutritionnelles.
    
    Fonctionnalités :
    - Normalisation des unités nutritionnelles
    - Calcul de valeurs manquantes
    - Détection d'allergènes intelligente
    - Amélioration de la qualité des données
    """
    
    def __init__(self):
        """Initialise l'enrichisseur avec ses bases de données de référence."""
        
        # Base de données d'allergènes communs avec leurs variantes
        self.allergen_keywords = {
            "gluten": {
                "keywords": ["gluten", "wheat", "blé", "froment", "épeautre", "kamut", "seigle", "orge", "avoine"],
                "ingredients": ["flour", "farine", "wheat", "barley", "rye", "oats"]
            },
            "milk": {
                "keywords": ["milk", "lait", "dairy", "lactose", "casein", "caséine", "whey", "lactosérum"],
                "ingredients": ["butter", "beurre", "cream", "crème", "cheese", "fromage", "yogurt", "yaourt"]
            },
            "eggs": {
                "keywords": ["egg", "œuf", "oeuf", "albumin", "albumine", "lecithin", "lécithine"],
                "ingredients": ["mayonnaise", "mayonnaise"]
            },
            "nuts": {
                "keywords": ["nuts", "noix", "almond", "amande", "walnut", "hazelnut", "noisette", "pecan", "cashew", "cajou"],
                "ingredients": ["marzipan", "massepain", "praline"]
            },
            "peanuts": {
                "keywords": ["peanut", "arachide", "groundnut", "cacahuète"],
                "ingredients": ["peanut oil", "huile d'arachide"]
            },
            "soy": {
                "keywords": ["soy", "soja", "tofu", "tempeh", "miso", "soybean", "edamame"],
                "ingredients": ["soy sauce", "sauce soja", "soy lecithin", "lécithine de soja"]
            },
            "fish": {
                "keywords": ["fish", "poisson", "salmon", "saumon", "tuna", "thon", "cod", "morue", "anchovy", "anchois"],
                "ingredients": ["fish sauce", "sauce poisson", "worcestershire"]
            },
            "shellfish": {
                "keywords": ["shellfish", "crevette", "shrimp", "crab", "crabe", "lobster", "homard", "mussel", "moule"],
                "ingredients": ["oyster sauce", "sauce huître"]
            },
            "sesame": {
                "keywords": ["sesame", "sésame", "tahini", "tahina"],
                "ingredients": ["sesame oil", "huile de sésame"]
            }
        }
        
        # Additifs alimentaires avec codes E
        self.additive_codes = {
            # Colorants
            "E100": "Curcumine", "E101": "Riboflavine", "E102": "Tartrazine",
            "E110": "Jaune orangé S", "E120": "Carmin", "E122": "Azorubine",
            
            # Conservateurs
            "E200": "Acide sorbique", "E202": "Sorbate de potassium", "E211": "Benzoate de sodium",
            "E220": "Anhydride sulfureux", "E250": "Nitrite de sodium", "E252": "Nitrate de potassium",
            
            # Antioxydants
            "E300": "Acide ascorbique", "E301": "Ascorbate de sodium", "E306": "Tocophérols",
            "E320": "BHA", "E321": "BHT",
            
            # Émulsifiants
            "E322": "Lécithines", "E471": "Mono- et diglycérides", "E472e": "Esters",
            
            # Édulcorants
            "E950": "Acésulfame K", "E951": "Aspartame", "E952": "Cyclamate", "E955": "Sucralose"
        }
        
        logger.info("Nutrition data enricher initialized")
    
    async def enrich_product_data(self, barcode_result: BarcodeResult) -> BarcodeResult:
        """
        Enrichit complètement les données d'un produit.
        
        Pipeline d'enrichissement :
        1. Normalisation des données nutritionnelles
        2. Extraction d'allergènes intelligente
        3. Identification des additifs
        4. Calcul de scores de qualité
        5. Amélioration des métadonnées
        """
        logger.info(
            "Enriching product data",
            barcode=barcode_result.barcode,
            product_name=barcode_result.product_name
        )
        
        # 1. Normalisation nutrition
        if barcode_result.nutrition_facts:
            barcode_result.nutrition_facts = await self.normalize_nutrition_facts(
                barcode_result.nutrition_facts
            )
        
        # 2. Extraction d'allergènes enrichie
        enhanced_allergens = await self.extract_allergens(
            barcode_result.ingredients or [],
            barcode_result.raw_data or {}
        )
        
        # Fusion avec allergènes existants
        existing_allergens = set(barcode_result.allergens or [])
        enhanced_allergens_set = set(enhanced_allergens)
        barcode_result.allergens = list(existing_allergens.union(enhanced_allergens_set))
        
        # 3. Identification des additifs
        enhanced_additives = await self._identify_additives(
            barcode_result.ingredients or [],
            barcode_result.additives or []
        )
        barcode_result.additives = enhanced_additives
        
        # 4. Calcul de scores nutritionnels
        nutrition_scores = await self._calculate_nutrition_scores(barcode_result.nutrition_facts or {})
        
        # 5. Amélioration des métadonnées
        enhanced_metadata = {
            "enrichment_version": "1.0",
            "nutrition_scores": nutrition_scores,
            "allergen_confidence": await self._calculate_allergen_confidence(barcode_result.allergens or []),
            "data_completeness": await self._calculate_data_completeness(barcode_result),
            "processing_timestamp": asyncio.get_event_loop().time()
        }
        
        # Fusion avec raw_data existant
        if barcode_result.raw_data:
            barcode_result.raw_data["wellix_enrichment"] = enhanced_metadata
        else:
            barcode_result.raw_data = {"wellix_enrichment": enhanced_metadata}
        
        # 6. Mise à jour de la confiance globale
        original_confidence = barcode_result.confidence
        enrichment_boost = min(0.2, nutrition_scores.get("completeness_score", 0) * 0.2)
        barcode_result.confidence = min(1.0, original_confidence + enrichment_boost)
        
        logger.info(
            "Product data enrichment completed",
            barcode=barcode_result.barcode,
            original_confidence=original_confidence,
            new_confidence=barcode_result.confidence,
            allergens_found=len(barcode_result.allergens),
            additives_found=len(barcode_result.additives)
        )
        
        return barcode_result
    
    async def normalize_nutrition_facts(self, raw_nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise les données nutritionnelles vers un format standard.
        
        Transformations :
        - Conversion d'unités (kJ -> kcal, mg -> g)
        - Calcul de valeurs manquantes
        - Validation de cohérence
        - Ajout de pourcentages AJR
        """
        normalized = {}
        
        # Copie des valeurs existantes
        for key, value in raw_nutrition.items():
            if value is not None and isinstance(value, (int, float)):
                normalized[key] = float(value)
        
        # Conversion kJ vers kcal si nécessaire
        if "energy_kj" in normalized and "calories" not in normalized:
            normalized["calories"] = normalized["energy_kj"] / 4.184
        
        # Conversion mg vers g pour sodium
        if "sodium_mg" in normalized and "sodium" not in normalized:
            normalized["sodium"] = normalized["sodium_mg"] / 1000
        
        # Calcul de calories si manquantes (4-4-9 rule)
        if "calories" not in normalized:
            protein = normalized.get("protein", 0)
            carbs = normalized.get("carbohydrates", 0)
            fat = normalized.get("total_fat", 0)
            
            if protein or carbs or fat:
                calculated_calories = (protein * 4) + (carbs * 4) + (fat * 9)
                normalized["calories"] = calculated_calories
                normalized["_calculated_calories"] = True
        
        # Validation de cohérence
        await self._validate_nutrition_consistency(normalized)
        
        # Calcul des pourcentages AJR (Apports Journaliers Recommandés)
        ajr_values = await self._calculate_daily_values(normalized)
        normalized.update(ajr_values)
        
        return normalized
    
    async def extract_allergens(self, ingredients: List[str], raw_data: Dict[str, Any]) -> List[str]:
        """
        Extraction intelligente d'allergènes depuis les ingrédients.
        
        Méthodes :
        - Analyse textuelle des ingrédients
        - Reconnaissance de patterns
        - Validation croisée avec données existantes
        """
        found_allergens = set()
        
        # Combinaison de tous les textes à analyser
        text_sources = []
        
        # Ingrédients principaux
        if ingredients:
            text_sources.extend(ingredients)
        
        # Texte d'ingrédients depuis raw_data
        if raw_data.get("ingredients_text"):
            text_sources.append(raw_data["ingredients_text"])
        
        if raw_data.get("ingredients_text_fr"):
            text_sources.append(raw_data["ingredients_text_fr"])
        
        # Analyse de chaque source
        combined_text = " ".join(text_sources).lower()
        
        for allergen, data in self.allergen_keywords.items():
            # Recherche de mots-clés directs
            for keyword in data["keywords"]:
                if keyword.lower() in combined_text:
                    found_allergens.add(allergen)
                    break
            
            # Recherche dans les ingrédients spécifiques
            for ingredient_pattern in data["ingredients"]:
                if ingredient_pattern.lower() in combined_text:
                    found_allergens.add(allergen)
                    break
        
        # Analyse par expressions régulières pour cas complexes
        allergen_patterns = {
            "gluten": r"(?:contains?\s+)?(?:wheat|gluten|blé|froment)",
            "milk": r"(?:contains?\s+)?(?:milk|lait|dairy|lactose)",
            "nuts": r"(?:contains?\s+)?(?:nuts?|noix|amandes?|noisettes?)"
        }
        
        for allergen, pattern in allergen_patterns.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                found_allergens.add(allergen)
        
        return list(found_allergens)
    
    async def _identify_additives(self, ingredients: List[str], existing_additives: List[str]) -> List[str]:
        """Identifie les additifs alimentaires dans les ingrédients."""
        found_additives = set(existing_additives)
        
        # Recherche de codes E
        combined_text = " ".join(ingredients).upper()
        e_pattern = r"E\d{3,4}[a-z]?"
        
        e_codes = re.findall(e_pattern, combined_text)
        
        for code in e_codes:
            if code in self.additive_codes:
                additive_name = f"{code} ({self.additive_codes[code]})"
                found_additives.add(additive_name)
            else:
                found_additives.add(code)
        
        # Recherche d'additifs par nom
        common_additives = [
            "lecithin", "lécithine", "citric acid", "acide citrique",
            "ascorbic acid", "acide ascorbique", "sodium benzoate",
            "benzoate de sodium", "potassium sorbate", "sorbate de potassium"
        ]
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            for additive in common_additives:
                if additive in ingredient_lower:
                    found_additives.add(additive.title())
        
        return list(found_additives)
    
    async def _calculate_nutrition_scores(self, nutrition_facts: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule des scores nutritionnels basés sur les recommandations."""
        scores = {}
        
        # Score de complétude (0-1)
        essential_nutrients = ["calories", "protein", "carbohydrates", "total_fat"]
        present_nutrients = sum(1 for nutrient in essential_nutrients if nutrient in nutrition_facts)
        scores["completeness_score"] = present_nutrients / len(essential_nutrients)
        
        # Score de qualité nutritionnelle simplifié (0-100)
        quality_score = 50  # Score de base
        
        # Bonus pour fibres
        if nutrition_facts.get("fiber", 0) >= 3:
            quality_score += 10
        
        # Bonus pour protéines
        if nutrition_facts.get("protein", 0) >= 10:
            quality_score += 10
        
        # Malus pour sodium élevé
        if nutrition_facts.get("sodium", 0) > 1.5:  # > 1.5g pour 100g
            quality_score -= 15
        
        # Malus pour sucres élevés
        if nutrition_facts.get("sugar", 0) > 15:  # > 15g pour 100g
            quality_score -= 10
        
        # Malus pour graisses saturées élevées
        if nutrition_facts.get("saturated_fat", 0) > 5:  # > 5g pour 100g
            quality_score -= 10
        
        scores["nutrition_quality"] = max(0, min(100, quality_score))
        
        return scores
    
    async def _validate_nutrition_consistency(self, nutrition: Dict[str, Any]):
        """Valide la cohérence des données nutritionnelles."""
        # Vérification que les graisses saturées <= graisses totales
        if ("saturated_fat" in nutrition and "total_fat" in nutrition and
            nutrition["saturated_fat"] > nutrition["total_fat"]):
            # Correction automatique
            nutrition["saturated_fat"] = nutrition["total_fat"]
            nutrition["_corrected_saturated_fat"] = True
        
        # Vérification que les sucres <= glucides totaux
        if ("sugar" in nutrition and "carbohydrates" in nutrition and
            nutrition["sugar"] > nutrition["carbohydrates"]):
            nutrition["sugar"] = nutrition["carbohydrates"]
            nutrition["_corrected_sugar"] = True
    
    async def _calculate_daily_values(self, nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les pourcentages des apports journaliers recommandés."""
        daily_values = {}
        
        # Valeurs de référence (pour 100g, basées sur un régime de 2000 kcal)
        reference_values = {
            "calories": 2000,
            "total_fat": 65,
            "saturated_fat": 20,
            "carbohydrates": 300,
            "fiber": 25,
            "protein": 50,
            "sodium": 2.3  # en grammes
        }
        
        for nutrient, reference in reference_values.items():
            if nutrient in nutrition:
                percentage = (nutrition[nutrient] / reference) * 100
                daily_values[f"{nutrient}_dv"] = round(percentage, 1)
        
        return daily_values
    
    async def _calculate_allergen_confidence(self, allergens: List[str]) -> float:
        """Calcule un score de confiance pour la détection d'allergènes."""
        if not allergens:
            return 1.0
        
        # Score de base élevé pour détection d'allergènes
        # (mieux vaut un faux positif qu'un faux négatif)
        confidence = 0.8
        
        # Bonus si détection multiple cohérente
        if len(allergens) > 1:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _calculate_data_completeness(self, barcode_result: BarcodeResult) -> float:
        """Calcule le score de complétude des données."""
        total_fields = 10
        present_fields = 0
        
        if barcode_result.product_name:
            present_fields += 1
        if barcode_result.brand:
            present_fields += 1
        if barcode_result.categories:
            present_fields += 1
        if barcode_result.nutrition_facts:
            present_fields += 1
        if barcode_result.ingredients:
            present_fields += 1
        if barcode_result.allergens:
            present_fields += 1
        if barcode_result.serving_size:
            present_fields += 1
        if barcode_result.image_urls:
            present_fields += 1
        if barcode_result.packaging:
            present_fields += 1
        if barcode_result.labels:
            present_fields += 1
        
        return present_fields / total_fields