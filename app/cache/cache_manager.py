"""
High-level cache manager with domain-specific caching strategies.
"""

from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import hashlib
import structlog

from app.cache.redis_client import redis_client, CacheKeys, CacheTTL
from app.models.health import UserHealthContext, AnalysisResult, ProductRecommendation

logger = structlog.get_logger(__name__)


class CacheManager:
    """High-level cache manager for domain-specific operations."""
    
    def __init__(self):
        self.redis = redis_client
    
    # User Health Context Caching
    async def get_user_health_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user health context."""
        key = CacheKeys.user_health_context(user_id)
        return await self.redis.get(key)
    
    async def set_user_health_context(self, user_id: str, context: Dict[str, Any]) -> bool:
        """Cache user health context."""
        key = CacheKeys.user_health_context(user_id)
        return await self.redis.set(key, context, ttl=CacheTTL.USER_HEALTH_CONTEXT)
    
    # Food Analysis Caching
    async def get_food_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get cached food analysis result."""
        key = CacheKeys.food_analysis(analysis_id)
        return await self.redis.get(key)
    
    async def set_food_analysis(self, analysis_id: str, analysis: Dict[str, Any]) -> bool:
        """Cache food analysis result."""
        key = CacheKeys.food_analysis(analysis_id)
        return await self.redis.set(key, analysis, ttl=CacheTTL.FOOD_ANALYSIS)
    
    # Product Data Caching
    async def get_product_data(self, barcode: str) -> Optional[Dict[str, Any]]:
        """Get cached product data."""
        key = CacheKeys.product_data(barcode)
        return await self.redis.get(key)
    
    async def set_product_data(self, barcode: str, product_data: Dict[str, Any]) -> bool:
        """Cache product data."""
        key = CacheKeys.product_data(barcode)
        return await self.redis.set(key, product_data, ttl=CacheTTL.PRODUCT_DATA)
    
    # OpenFoodFacts Caching
    async def get_openfoodfacts_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        """Get cached OpenFoodFacts product data."""
        key = CacheKeys.openfoodfacts_product(barcode)
        return await self.redis.get(key)
    
    async def set_openfoodfacts_product(self, barcode: str, product_data: Dict[str, Any]) -> bool:
        """Cache OpenFoodFacts product data."""
        key = CacheKeys.openfoodfacts_product(barcode)
        return await self.redis.set(key, product_data, ttl=CacheTTL.OPENFOODFACTS_DATA)
    
    # Recommendations Caching
    async def get_recommendations(self, analysis_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached product recommendations."""
        key = CacheKeys.recommendations(analysis_id)
        return await self.redis.get(key)
    
    async def set_recommendations(self, analysis_id: str, recommendations: List[Dict[str, Any]]) -> bool:
        """Cache product recommendations."""
        key = CacheKeys.recommendations(analysis_id)
        return await self.redis.set(key, recommendations, ttl=CacheTTL.RECOMMENDATIONS)
    
    # OCR Result Caching
    async def get_ocr_result(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """Get cached OCR result by image hash."""
        image_hash = hashlib.sha256(image_data).hexdigest()
        key = CacheKeys.ocr_result(image_hash)
        return await self.redis.get(key)
    
    async def set_ocr_result(self, image_data: bytes, ocr_result: Dict[str, Any]) -> bool:
        """Cache OCR result by image hash."""
        image_hash = hashlib.sha256(image_data).hexdigest()
        key = CacheKeys.ocr_result(image_hash)
        return await self.redis.set(key, ocr_result, ttl=CacheTTL.OCR_RESULT)
    
    # Chat Context Caching
    async def get_chat_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached chat context."""
        key = CacheKeys.chat_context(session_id)
        return await self.redis.get(key)
    
    async def set_chat_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Cache chat context."""
        key = CacheKeys.chat_context(session_id)
        return await self.redis.set(key, context, ttl=CacheTTL.CHAT_CONTEXT)
    
    # Rate Limiting
    async def check_rate_limit(self, user_id: str, endpoint: str, limit: int = 100) -> bool:
        """Check if user has exceeded rate limit for endpoint."""
        key = CacheKeys.rate_limit(user_id, endpoint)
        current_count = await self.redis.get(key) or 0
        
        if isinstance(current_count, str):
            current_count = int(current_count)
        
        return current_count < limit
    
    async def increment_rate_limit(self, user_id: str, endpoint: str) -> Optional[int]:
        """Increment rate limit counter for user and endpoint."""
        key = CacheKeys.rate_limit(user_id, endpoint)
        
        # Check if key exists
        if not await self.redis.exists(key):
            # First request, set with TTL
            await self.redis.set(key, 1, ttl=CacheTTL.RATE_LIMIT_WINDOW)
            return 1
        else:
            # Increment existing counter
            return await self.redis.increment(key)
    
    # Cache Invalidation
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user."""
        keys_to_delete = [
            CacheKeys.user_session(user_id),
            CacheKeys.user_health_context(user_id),
        ]
        
        await self.redis.delete(*keys_to_delete)
        logger.info(f"Invalidated cache for user {user_id}")
    
    async def invalidate_analysis_cache(self, analysis_id: str):
        """Invalidate cache entries for a specific analysis."""
        keys_to_delete = [
            CacheKeys.food_analysis(analysis_id),
            CacheKeys.recommendations(analysis_id),
        ]
        
        await self.redis.delete(*keys_to_delete)
        logger.info(f"Invalidated cache for analysis {analysis_id}")
    
    # Cache Statistics
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and health metrics."""
        try:
            info = await self.redis.client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)


# Global cache manager instance
cache_manager = CacheManager()
