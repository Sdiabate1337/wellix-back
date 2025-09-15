#!/usr/bin/env python3
"""
Startup script for the Wellix AI Food Analysis Backend.
Handles database initialization, migrations, and server startup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn
import structlog
from alembic.config import Config
from alembic import command
from sqlalchemy import text

from app.core.config import settings
from app.db.database import async_engine, sync_engine
from app.cache.redis_client import redis_client

logger = structlog.get_logger(__name__)


async def check_database_connection():
    """Check if database is accessible."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def check_redis_connection():
    """Check if Redis is accessible."""
    try:
        await redis_client.ping()
        logger.info("Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


def run_migrations():
    """Run Alembic database migrations."""
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


async def initialize_services():
    """Initialize all required services."""
    logger.info("Initializing Wellix backend services...")
    
    # Check database connection
    if not await check_database_connection():
        logger.error("Cannot start without database connection")
        return False
    
    # Check Redis connection
    if not await check_redis_connection():
        logger.warning("Redis connection failed - caching will be disabled")
    
    # Skip migrations for now
    logger.info("Skipping migrations for debugging")
    
    logger.info("All services initialized successfully")
    return True


def start_development_server():
    """Start the development server with hot reload."""
    logger.info("Starting Wellix backend in development mode...")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="info",
        access_log=True,
        use_colors=True,
        loop="asyncio"
    )


def start_production_server():
    """Start the production server."""
    logger.info("Starting Wellix backend in production mode...")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="warning",
        access_log=False,
        loop="asyncio"
    )


async def main():
    """Main startup function."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger.info("Starting Wellix AI Food Analysis Backend")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize services
    if not await initialize_services():
        logger.error("Service initialization failed")
        sys.exit(1)
    
    # Start appropriate server
    if settings.environment == "development":
        start_development_server()
    else:
        start_production_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)
