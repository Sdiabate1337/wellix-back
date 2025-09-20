"""
Semantic Cache pour optimisation des coûts LLM.

Implémente un cache intelligent basé sur la similarité sémantique
pour éviter les requêtes LLM redondantes et réduire les coûts.
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog
import pickle
import re
from pathlib import Path

from .interfaces import ISemanticCache, LLMResult, LLMServiceException

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Entrée du cache sémantique."""
    
    input_hash: str
    input_data: Dict[str, Any]
    result: LLMResult
    embedding: Optional[List[float]] = None
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # TTL et invalidation
    ttl_hours: int = 24  # Par défaut 24h
    is_valid: bool = True
    
    @property
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        if not self.is_valid:
            return True
        
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    @property
    def age_hours(self) -> float:
        """Âge de l'entrée en heures."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def access(self):
        """Marque l'entrée comme accédée."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class SemanticCache(ISemanticCache):
    """
    Cache sémantique intelligent pour résultats LLM.
    
    Fonctionnalités:
    - Hashing rapide pour correspondances exactes
    - Embeddings pour similarité sémantique
    - TTL configurable par type de données
    - Éviction automatique (LRU + âge)
    - Persistance sur disque optionnelle
    - Métriques de cache hit/miss
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 max_entries: int = 10000,
                 default_ttl_hours: int = 24,
                 enable_persistence: bool = True,
                 similarity_threshold: float = 0.95):
        
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours
        self.enable_persistence = enable_persistence
        self.similarity_threshold = similarity_threshold
        
        # Cache en mémoire
        self.cache: Dict[str, CacheEntry] = {}
        self.similarity_index: Dict[str, List[str]] = {}  # hash -> similar hashes
        
        # Métriques
        self.hits = 0
        self.misses = 0
        self.similarity_hits = 0
        
        # Configuration du répertoire cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / "cache" / "llm_semantic"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Chargement du cache persistant
        if self.enable_persistence:
            asyncio.create_task(self._load_cache())
        
        logger.info(
            "Semantic Cache initialized",
            cache_dir=str(self.cache_dir),
            max_entries=max_entries,
            similarity_threshold=similarity_threshold,
            persistence_enabled=enable_persistence
        )
    
    async def get_similar_analysis(self, input_data: Dict[str, Any], 
                                 threshold: float = None) -> Optional[LLMResult]:
        """
        Recherche une analyse similaire dans le cache.
        
        Args:
            input_data: Données d'entrée
            threshold: Seuil de similarité (utilise default si None)
            
        Returns:
            Optional[LLMResult]: Résultat similaire ou None
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        try:
            # 1. Recherche correspondance exacte
            input_hash = self._hash_input(input_data)
            
            if input_hash in self.cache:
                entry = self.cache[input_hash]
                
                if not entry.is_expired:
                    entry.access()
                    self.hits += 1
                    
                    # Clone le résultat avec indication de cache hit
                    cached_result = self._clone_result_with_cache_info(entry.result, 1.0)
                    
                    logger.debug(
                        "Exact cache hit",
                        input_hash=input_hash[:8],
                        age_hours=entry.age_hours,
                        access_count=entry.access_count
                    )
                    
                    return cached_result
                else:
                    # Entrée expirée, la supprime
                    await self._remove_entry(input_hash)
            
            # 2. Recherche par similarité sémantique
            similar_result = await self._find_similar_entry(input_data, threshold)
            
            if similar_result:
                self.similarity_hits += 1
                return similar_result
            
            # 3. Aucune correspondance trouvée
            self.misses += 1
            return None
            
        except Exception as e:
            logger.error(
                "Cache lookup failed",
                error=str(e),
                input_keys=list(input_data.keys())
            )
            self.misses += 1
            return None
    
    async def store_analysis(self, input_data: Dict[str, Any], result: LLMResult) -> None:
        """
        Stocke une analyse dans le cache.
        
        Args:
            input_data: Données d'entrée
            result: Résultat à stocker
        """
        try:
            input_hash = self._hash_input(input_data)
            
            # Détermine TTL selon le type de tâche
            ttl_hours = self._determine_ttl(result.task_type)
            
            # Crée l'entrée cache
            entry = CacheEntry(
                input_hash=input_hash,
                input_data=input_data,
                result=result,
                ttl_hours=ttl_hours
            )
            
            # Calcule embedding pour recherche sémantique
            if await self._should_compute_embedding(input_data):
                entry.embedding = await self._compute_embedding(input_data)
            
            # Stocke dans le cache
            await self._store_entry(input_hash, entry)
            
            # Éviction si nécessaire
            if len(self.cache) > self.max_entries:
                await self._evict_entries()
            
            # Sauvegarde persistante
            if self.enable_persistence:
                asyncio.create_task(self._save_entry(input_hash, entry))
            
            logger.debug(
                "Analysis stored in cache",
                input_hash=input_hash[:8],
                task_type=result.task_type,
                ttl_hours=ttl_hours,
                has_embedding=entry.embedding is not None
            )
            
        except Exception as e:
            logger.error(
                "Cache storage failed",
                error=str(e),
                task_type=result.task_type
            )
    
    async def calculate_similarity(self, data1: Dict[str, Any], 
                                 data2: Dict[str, Any]) -> float:
        """
        Calcule la similarité entre deux jeux de données.
        
        Args:
            data1: Premier jeu de données
            data2: Deuxième jeu de données
            
        Returns:
            float: Score de similarité (0-1)
        """
        try:
            # 1. Similarité structurelle (clés)
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            
            if not keys1 and not keys2:
                return 1.0
            if not keys1 or not keys2:
                return 0.0
            
            key_similarity = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
            
            # 2. Similarité des valeurs
            value_similarities = []
            
            for key in keys1.intersection(keys2):
                val1 = data1[key]
                val2 = data2[key]
                
                val_sim = await self._calculate_value_similarity(val1, val2)
                value_similarities.append(val_sim)
            
            value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.0
            
            # 3. Score global (50% structure, 50% valeurs)
            overall_similarity = (key_similarity * 0.5) + (value_similarity * 0.5)
            
            return overall_similarity
            
        except Exception as e:
            logger.error("Similarity calculation failed", error=str(e))
            return 0.0
    
    def _hash_input(self, input_data: Dict[str, Any]) -> str:
        """Génère un hash des données d'entrée."""
        # Normalise les données pour hash cohérent
        normalized = self._normalize_data(input_data)
        
        # Sérialise de manière déterministe
        serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        
        # Hash SHA-256
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise les données pour comparaison cohérente."""
        normalized = {}
        
        for key, value in data.items():
            # Normalise les clés
            norm_key = key.lower().strip()
            
            # Normalise les valeurs
            if isinstance(value, str):
                # Normalise les strings
                norm_value = re.sub(r'\s+', ' ', value.lower().strip())
            elif isinstance(value, list):
                # Trie les listes pour cohérence
                norm_value = sorted([
                    v.lower().strip() if isinstance(v, str) else v 
                    for v in value
                ])
            elif isinstance(value, dict):
                # Récursion pour dictionnaires
                norm_value = self._normalize_data(value)
            else:
                norm_value = value
            
            normalized[norm_key] = norm_value
        
        return normalized
    
    def _determine_ttl(self, task_type) -> int:
        """Détermine TTL selon le type de tâche."""
        # TTL différents selon la criticité
        ttl_mapping = {
            "nutrition_analysis": 72,      # 3 jours (stable)
            "allergen_detection": 168,     # 7 jours (très stable)
            "health_impact_assessment": 48, # 2 jours (peut évoluer)
            "ingredient_parsing": 168,     # 7 jours (stable)
            "dietary_compliance": 72,      # 3 jours
            "product_categorization": 168, # 7 jours (stable)
            "claims_validation": 24,       # 1 jour (réglementations évoluent)
            "recipe_analysis": 72,         # 3 jours
            "nutritional_comparison": 48   # 2 jours
        }
        
        task_str = str(task_type).lower()
        return ttl_mapping.get(task_str, self.default_ttl_hours)
    
    async def _should_compute_embedding(self, input_data: Dict[str, Any]) -> bool:
        """Détermine si l'embedding doit être calculé."""
        # Compute embedding seulement pour données textuelles significatives
        text_fields = ['product_name', 'ingredients', 'description', 'marketing_claims']
        
        has_significant_text = any(
            key in input_data and isinstance(input_data[key], str) and len(input_data[key]) > 10
            for key in text_fields
        )
        
        return has_significant_text
    
    async def _compute_embedding(self, input_data: Dict[str, Any]) -> List[float]:
        """
        Calcule l'embedding des données d'entrée.
        
        Note: Implémentation simplifiée. En production, utiliserait
        un modèle d'embedding comme sentence-transformers ou OpenAI embeddings.
        """
        # Concatène les champs textuels significatifs
        text_parts = []
        
        text_fields = ['product_name', 'ingredients', 'description', 'marketing_claims']
        for field in text_fields:
            if field in input_data:
                value = input_data[field]
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    text_parts.extend(str(v) for v in value)
        
        text = " ".join(text_parts).lower()
        
        # Embedding simple basé sur hashing (remplacer par vrai embedding en prod)
        # Divise le texte en tokens et hash chaque token
        tokens = re.findall(r'\w+', text)
        
        # Crée un vecteur de taille fixe (128 dimensions)
        embedding = [0.0] * 128
        
        for token in tokens:
            token_hash = hash(token) % 128
            embedding[token_hash] += 1.0
        
        # Normalise le vecteur
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def _find_similar_entry(self, input_data: Dict[str, Any], 
                                threshold: float) -> Optional[LLMResult]:
        """Trouve une entrée similaire dans le cache."""
        if not self.cache:
            return None
        
        input_embedding = await self._compute_embedding(input_data)
        if not input_embedding:
            return None
        
        best_similarity = 0.0
        best_entry = None
        
        for entry in self.cache.values():
            if entry.is_expired or not entry.embedding:
                continue
            
            # Calcule similarité cosinus
            similarity = self._cosine_similarity(input_embedding, entry.embedding)
            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
        
        if best_entry:
            best_entry.access()
            cached_result = self._clone_result_with_cache_info(
                best_entry.result, best_similarity
            )
            
            logger.debug(
                "Semantic cache hit",
                similarity=best_similarity,
                threshold=threshold,
                age_hours=best_entry.age_hours
            )
            
            return cached_result
        
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux vecteurs."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _calculate_value_similarity(self, val1: Any, val2: Any) -> float:
        """Calcule la similarité entre deux valeurs."""
        if type(val1) != type(val2):
            return 0.0
        
        if isinstance(val1, str):
            # Similarité textuelle simple
            if val1.lower() == val2.lower():
                return 1.0
            
            # Jaccard similarity pour mots
            words1 = set(re.findall(r'\w+', val1.lower()))
            words2 = set(re.findall(r'\w+', val2.lower()))
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
        
        elif isinstance(val1, (int, float)):
            # Similarité numérique
            if val1 == val2:
                return 1.0
            
            # Différence relative
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return 1.0
            
            relative_diff = abs(val1 - val2) / max_val
            return max(0.0, 1.0 - relative_diff)
        
        elif isinstance(val1, list):
            # Similarité de listes
            set1 = set(str(v).lower() for v in val1)
            set2 = set(str(v).lower() for v in val2)
            
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            
            return len(intersection) / len(union)
        
        elif isinstance(val1, dict):
            # Similarité récursive pour dictionnaires
            return await self.calculate_similarity(val1, val2)
        
        else:
            # Égalité simple pour autres types
            return 1.0 if val1 == val2 else 0.0
    
    def _clone_result_with_cache_info(self, result: LLMResult, 
                                    similarity: float) -> LLMResult:
        """Clone un résultat avec informations de cache."""
        from copy import deepcopy
        
        cloned = deepcopy(result)
        cloned.cache_hit = True
        cloned.cache_similarity = similarity
        
        return cloned
    
    async def _store_entry(self, input_hash: str, entry: CacheEntry):
        """Stocke une entrée dans le cache mémoire."""
        self.cache[input_hash] = entry
    
    async def _remove_entry(self, input_hash: str):
        """Supprime une entrée du cache."""
        if input_hash in self.cache:
            del self.cache[input_hash]
        
        # Supprime aussi du fichier si persistance activée
        if self.enable_persistence:
            cache_file = self.cache_dir / f"{input_hash}.cache"
            if cache_file.exists():
                cache_file.unlink()
    
    async def _evict_entries(self):
        """Éviction d'entrées selon politique LRU + âge."""
        if len(self.cache) <= self.max_entries:
            return
        
        # Calcule scores d'éviction (plus bas = évincé en premier)
        eviction_scores = {}
        
        for input_hash, entry in self.cache.items():
            if entry.is_expired:
                eviction_scores[input_hash] = -1  # Évince immédiatement
            else:
                # Score basé sur age et fréquence d'accès
                age_penalty = entry.age_hours / 24  # Pénalité par jour
                access_bonus = entry.access_count * 0.1
                
                score = access_bonus - age_penalty
                eviction_scores[input_hash] = score
        
        # Trie par score croissant et évince les plus bas
        sorted_entries = sorted(eviction_scores.items(), key=lambda x: x[1])
        
        entries_to_evict = len(self.cache) - self.max_entries + 100  # Évince 100 de plus pour éviter évictions fréquentes
        
        for i in range(min(entries_to_evict, len(sorted_entries))):
            input_hash = sorted_entries[i][0]
            await self._remove_entry(input_hash)
        
        logger.info(
            "Cache eviction completed",
            evicted_count=entries_to_evict,
            remaining_entries=len(self.cache)
        )
    
    async def _load_cache(self):
        """Charge le cache depuis le disque."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            loaded_count = 0
            
            for cache_file in cache_files[:self.max_entries]:  # Limite le chargement
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if not entry.is_expired:
                        self.cache[entry.input_hash] = entry
                        loaded_count += 1
                    else:
                        # Supprime fichier expiré
                        cache_file.unlink()
                
                except Exception as e:
                    logger.warning(
                        "Failed to load cache entry",
                        file=cache_file.name,
                        error=str(e)
                    )
                    # Supprime fichier corrompu
                    cache_file.unlink()
            
            logger.info(
                "Cache loaded from disk",
                loaded_entries=loaded_count,
                total_files=len(cache_files)
            )
            
        except Exception as e:
            logger.error("Cache loading failed", error=str(e))
    
    async def _save_entry(self, input_hash: str, entry: CacheEntry):
        """Sauvegarde une entrée sur disque."""
        try:
            cache_file = self.cache_dir / f"{input_hash}.cache"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        
        except Exception as e:
            logger.warning(
                "Failed to save cache entry",
                input_hash=input_hash[:8],
                error=str(e)
            )
    
    @property
    def hit_rate(self) -> float:
        """Taux de succès du cache."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    @property
    def similarity_hit_rate(self) -> float:
        """Taux de succès par similarité."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.similarity_hits / total
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "total_entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "similarity_hits": self.similarity_hits,
            "hit_rate": self.hit_rate,
            "similarity_hit_rate": self.similarity_hit_rate,
            "expired_entries": sum(1 for entry in self.cache.values() if entry.is_expired)
        }