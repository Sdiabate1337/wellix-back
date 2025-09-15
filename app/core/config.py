"""
Core configuration management using Pydantic Settings.
Follows 12-factor app principles for environment-based configuration.
"""

from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Wellix AI Food Analysis API"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_url_sync: str = Field(..., env="DATABASE_URL_SYNC")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # AI Services
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    
    # Google Cloud
    google_application_credentials: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    google_cloud_project: Optional[str] = Field(default=None, env="GOOGLE_CLOUD_PROJECT")
    
    # Pinecone
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="wellix-products", env="PINECONE_INDEX_NAME")
    
    # External APIs
    openfoodfacts_user_agent: str = Field(
        default="Wellix/1.0 (contact@wellix.com)",
        env="OPENFOODFACTS_USER_AGENT"
    )
    
    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/1", env="CELERY_RESULT_BACKEND")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment values."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentConfig(Settings):
    """Development environment configuration."""
    debug: bool = True
    log_level: str = "DEBUG"


class ProductionConfig(Settings):
    """Production environment configuration."""
    debug: bool = False
    log_level: str = "WARNING"
    
    @validator("secret_key")
    def validate_secret_key_production(cls, v):
        """Ensure secret key is secure in production."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters in production")
        return v


def get_settings() -> Settings:
    """Factory function to get environment-specific settings."""
    env = settings.environment.lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "development":
        return DevelopmentConfig()
    else:
        return Settings()
