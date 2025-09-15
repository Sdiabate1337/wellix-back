"""
Redis client configuration and connection management.
Provides caching utilities for the Wellix application.
"""

import json
import pickle
from typing import Any, Optional, Union, Dict, List
from datetime import timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and error handling."""
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection pool."""
        try:
            self.pool = ConnectionPool.from_url(
                settings.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
            )
            
            self.client = redis.Redis(
                connection_pool=self.pool,
                decode_responses=False,  # We'll handle encoding manually
            )
            
            # Test connection
            await self.client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connections gracefully."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Redis connections closed")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with automatic deserialization."""
        try:
            if not self.client:
                await self.connect()
            
            value = await self.client.get(key)
            if value is None:
                return None
            
            # Try JSON first, then pickle
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(value)
                
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        serialize_json: bool = True
    ) -> bool:
        """Set value in Redis with automatic serialization."""
        try:
            if not self.client:
                await self.connect()
            
            # Serialize value
            if serialize_json:
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    serialized_value = pickle.dumps(value)
            else:
                serialized_value = pickle.dumps(value)
            
            # Set with TTL
            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                await self.client.setex(key, ttl, serialized_value)
            else:
                await self.client.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis."""
        try:
            if not self.client:
                await self.connect()
            
            return await self.client.delete(*keys)
            
        except Exception as e:
            logger.error(f"Redis DELETE error for keys {keys}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            if not self.client:
                await self.connect()
            
            return bool(await self.client.exists(key))
            
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """Set expiration time for a key."""
        try:
            if not self.client:
                await self.connect()
            
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            
            return bool(await self.client.expire(key, ttl))
            
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in Redis."""
        try:
            if not self.client:
                await self.connect()
            
            return await self.client.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Redis INCREMENT error for key {key}: {e}")
            return None
    
    async def ping(self) -> bool:
        """Test la connexion au serveur Redis."""
        try:
            if not self.client:
                await self.connect()
            response = await self.client.ping()
            return response is True
        except Exception as e:
            logger.error(f"Redis PING error: {e}")
            return False
    
    async def get_hash(self, key: str, field: str) -> Optional[Any]:
        """Get hash field value."""
        try:
            if not self.client:
                await self.connect()
            
            value = await self.client.hget(key, field)
            if value is None:
                return None
            
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(value)
                
        except Exception as e:
            logger.error(f"Redis HGET error for key {key}, field {field}: {e}")
            return None
    
    async def set_hash(self, key: str, field: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set hash field value."""
        try:
            if not self.client:
                await self.connect()
            
            # Serialize value
            try:
                serialized_value = json.dumps(value, default=str)
            except (TypeError, ValueError):
                serialized_value = pickle.dumps(value)
            
            await self.client.hset(key, field, serialized_value)
            
            if ttl:
                await self.client.expire(key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis HSET error for key {key}, field {field}: {e}")
            return False


# Global Redis client instance
redis_client = RedisClient()


# Cache key generators
class CacheKeys:
    """Cache key generators for different data types."""
    
    @staticmethod
    def user_session(user_id: str) -> str:
        return f"session:user:{user_id}"
    
    @staticmethod
    def user_health_context(user_id: str) -> str:
        return f"health:context:{user_id}"
    
    @staticmethod
    def food_analysis(analysis_id: str) -> str:
        return f"analysis:{analysis_id}"
    
    @staticmethod
    def product_data(barcode: str) -> str:
        return f"product:barcode:{barcode}"
    
    @staticmethod
    def openfoodfacts_product(barcode: str) -> str:
        return f"off:product:{barcode}"
    
    @staticmethod
    def recommendations(analysis_id: str) -> str:
        return f"recommendations:{analysis_id}"
    
    @staticmethod
    def chat_context(session_id: str) -> str:
        return f"chat:context:{session_id}"
    
    @staticmethod
    def rate_limit(user_id: str, endpoint: str) -> str:
        return f"rate_limit:{endpoint}:{user_id}"
    
    @staticmethod
    def ocr_result(image_hash: str) -> str:
        return f"ocr:result:{image_hash}"


# Cache TTL constants (in seconds)
class CacheTTL:
    """Cache TTL constants for different data types."""
    
    USER_SESSION = 3600  # 1 hour
    USER_HEALTH_CONTEXT = 3600  # 1 hour
    FOOD_ANALYSIS = 86400  # 24 hours
    PRODUCT_DATA = 604800  # 1 week
    OPENFOODFACTS_DATA = 604800  # 1 week
    RECOMMENDATIONS = 7200  # 2 hours
    CHAT_CONTEXT = 3600  # 1 hour
    OCR_RESULT = 86400  # 24 hours
    RATE_LIMIT_WINDOW = 3600  # 1 hour
