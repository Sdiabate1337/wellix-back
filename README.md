# Wellix AI Food Analysis Backend

A sophisticated FastAPI-based backend for AI-powered food analysis and health recommendations.

## Features

- **Multi-Profile Health Analysis**: Support for diabetes, hypertension, heart disease, kidney disease, obesity, and general wellness
- **AI-Powered Chat**: Context-aware conversations using OpenAI GPT-4, Anthropic Claude, and OpenRouter
- **OCR Integration**: Google Vision API for nutrition label extraction
- **LangGraph Workflows**: Orchestrated food analysis pipeline
- **Real-time Updates**: WebSocket support for streaming responses
- **Caching Layer**: Redis-based performance optimization
- **Authentication**: JWT-based security with rate limiting

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker (optional)

### Environment Setup

1. Copy environment variables:
```bash
cp .env.example .env
```

2. Update `.env` with your API keys and database credentials

### Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start services with Docker:
```bash
docker-compose up postgres redis -d
```

3. Run migrations:
```bash
alembic upgrade head
```

4. Start development server:
```bash
python start.py
```

### Production Deployment

1. Build and run with Docker:
```bash
docker-compose up -d
```

## API Documentation

Once running, visit:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Token refresh
- `GET /api/v1/auth/profile` - User profile
- `POST /api/v1/auth/logout` - User logout

### Health Profiles
- `POST /api/v1/health/profiles` - Create health profile
- `GET /api/v1/health/profiles` - Get user profiles
- `PUT /api/v1/health/profiles/{id}` - Update profile
- `DELETE /api/v1/health/profiles/{id}` - Delete profile
- `GET /api/v1/health/context` - Get health context
- `PUT /api/v1/health/context` - Update health context
- `GET /api/v1/health/recommendations` - Get recommendations

### Food Analysis
- `POST /api/v1/analysis/scan-food` - Analyze food image
- `GET /api/v1/analysis/{id}` - Get analysis results
- `GET /api/v1/analysis/history` - Analysis history
- `POST /api/v1/analysis/validate-image` - Validate food image

### AI Chat
- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/chat/sessions` - Get chat sessions
- `POST /api/v1/chat/sessions` - Create chat session
- `GET /api/v1/chat/sessions/{id}/messages` - Get messages
- `WebSocket /api/v1/ws/chat/{session_id}` - Real-time chat

## Architecture

### Core Components

- **FastAPI Application**: High-performance async web framework
- **PostgreSQL**: Primary database with SQLAlchemy ORM
- **Redis**: Caching and session storage
- **LangGraph**: Workflow orchestration for food analysis
- **AI Services**: OpenAI, Anthropic, OpenRouter integrations

### Health Analyzers

Multi-profile health analysis system supporting:

- **Diabetes**: Glycemic control, carbohydrate analysis
- **Hypertension**: Sodium monitoring, DASH diet principles
- **Heart Disease**: Heart-healthy fats, cholesterol management
- **Kidney Disease**: Protein, phosphorus, potassium monitoring
- **Obesity**: Calorie balance, portion control
- **General Wellness**: Balanced nutrition recommendations

### Workflow Pipeline

1. **Image Processing**: OCR extraction with Google Vision API
2. **Data Enrichment**: OpenFoodFacts product lookup
3. **Nutrition Analysis**: Structured data parsing
4. **Health Analysis**: Multi-profile scoring and recommendations
5. **AI Chat Context**: Contextual conversation preparation

## Configuration

### Environment Variables

Key configuration options:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/wellix
SYNC_DATABASE_URL=postgresql://user:pass@localhost/wellix

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Services
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENROUTER_API_KEY=your-openrouter-key

# Google Vision
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# OpenFoodFacts
OPENFOODFACTS_USER_AGENT=YourApp/1.0
```

## Development

### Project Structure

```
backend/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # Core utilities
│   ├── db/              # Database models
│   ├── services/        # Business logic
│   ├── workflows/       # LangGraph workflows
│   ├── cache/           # Caching layer
│   └── models/          # Pydantic models
├── alembic/             # Database migrations
├── tests/               # Test suite
└── start.py             # Startup script
```

### Adding New Health Analyzers

1. Create analyzer in `app/services/health_analyzers/`
2. Extend `ProfileType` enum in `app/models/health.py`
3. Register in `analyzer_factory.py`
4. Add database migration for new profile type

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_health_analyzers.py
```

## Deployment

### Docker Production

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f backend

# Scale services
docker-compose up -d --scale backend=3
```

### Health Monitoring

- Health endpoint: `/health`
- Metrics: Prometheus-compatible (future)
- Logging: Structured JSON logs with structlog

## Security

- JWT authentication with refresh tokens
- Rate limiting per user tier
- Input validation with Pydantic
- SQL injection protection with SQLAlchemy
- CORS configuration
- Environment-based secrets

## Performance

- Async/await throughout
- Redis caching with smart TTLs
- Database connection pooling
- Efficient SQL queries with proper indexing
- Background task processing (Celery ready)

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

Proprietary - Wellix AI Food Analysis Platform
