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

**Base URL:** `http://localhost:8000/api/v1`

### 🔐 Authentication (`/auth`)

#### Register User
```bash
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe"
  }
}
```

#### Login User
```bash
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

#### Refresh Token
```bash
POST /api/v1/auth/refresh
Authorization: Bearer <refresh_token>
```

#### Logout
```bash
POST /api/v1/auth/logout
Authorization: Bearer <access_token>
```

### 🏥 Health Profiles (`/health`)

#### Create Health Profile
```bash
POST /api/v1/health/profiles
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "profile_type": "diabetes",
  "severity": "moderate",
  "restrictions": ["sugar", "refined_carbs"],
  "goals": ["blood_sugar_control", "weight_management"],
  "medications": ["metformin"],
  "is_primary": true,
  "target_values": {
    "hba1c": 7.0,
    "fasting_glucose": 100.0
  }
}
```

#### Get Health Profiles
```bash
GET /api/v1/health/profiles
Authorization: Bearer <access_token>
```

#### Update Health Context
```bash
PUT /api/v1/health/context
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "age_group": "adult",
  "activity_level": "moderate",
  "weight_goals": "maintain",
  "height_cm": 175.0,
  "weight_kg": 70.0,
  "allergies": ["nuts", "dairy"],
  "dietary_preferences": ["vegetarian"],
  "preferred_language": "fr"
}
```

#### Get Health Context
```bash
GET /api/v1/health/context
Authorization: Bearer <access_token>
```

#### Get Health Recommendations
```bash
GET /api/v1/health/recommendations
Authorization: Bearer <access_token>
```

### 🔬 Food Analysis (`/analysis`)

#### Scan Food Product (Principal Endpoint)
```bash
POST /api/v1/analysis/scan-food
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

--form 'image=@/path/to/food_image.jpg'
--form 'barcode=3017620422003'
--form 'integration_level=expert_analysis'
```

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "product": {
    "name": "Nutella",
    "brand": "Ferrero",
    "barcode": "3017620422003"
  },
  "nutrition": {
    "energy_kcal": 539,
    "fat": 30.9,
    "saturated_fat": 10.6,
    "carbohydrates": 57.5,
    "sugars": 56.3,
    "fiber": 1.9,
    "protein": 6.3,
    "salt": 0.107
  },
  "health_analysis": {
    "diabetes": {
      "score": 2,
      "risk_level": "high",
      "recommendations": [
        "Évitez ce produit en raison de sa très haute teneur en sucres",
        "Choisissez des alternatives sans sucre ajouté"
      ]
    }
  },
  "overall_score": 2,
  "recommendation": "avoid",
  "ai_insights": "Ce produit contient 56.3g de sucres pour 100g..."
}
```

#### Analyze Nutrition Data
```bash
POST /api/v1/analysis/analyze-nutrition
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "nutrition_data": {
    "energy_kcal": 350,
    "carbohydrates": 45.0,
    "sugars": 5.0,
    "fiber": 8.0,
    "protein": 12.0,
    "fat": 15.0,
    "salt": 1.2
  },
  "product_name": "Whole grain bread"
}
```

#### Get Analysis History
```bash
GET /api/v1/analysis/history?limit=10&offset=0
Authorization: Bearer <access_token>
```

#### Get Specific Analysis
```bash
GET /api/v1/analysis/{analysis_id}
Authorization: Bearer <access_token>
```

#### Validate Food Image
```bash
POST /api/v1/analysis/validate-image
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

--form 'image=@/path/to/image.jpg'
```

### 💬 AI Chat (`/chat`)

#### Send Chat Message
```bash
POST /api/v1/chat/message
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "message": "Est-ce que je peux manger ce produit avec mon diabète ?",
  "context": {
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

#### Get Chat Sessions
```bash
GET /api/v1/chat/sessions
Authorization: Bearer <access_token>
```

#### WebSocket Chat
```bash
ws://localhost:8000/api/v1/ws/chat/{session_id}?token=<access_token>
```

### 🔧 LLM Configuration (`/llm`)

#### Get Available Models
```bash
GET /api/v1/llm/models
Authorization: Bearer <access_token>
```

#### Update LLM Configuration
```bash
PUT /api/v1/llm/config
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "default_model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

## 🚀 Complete User Workflow

### 1. User Registration & Authentication
```bash
# 1. Register a new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!",
    "first_name": "Test",
    "last_name": "User"
  }'

# 2. Login to get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!"
  }'
```

### 2. Health Profile Setup
```bash
# 3. Create diabetes health profile
curl -X POST "http://localhost:8000/api/v1/health/profiles" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "profile_type": "diabetes",
    "severity": "moderate",
    "restrictions": ["sugar", "refined_carbs"],
    "goals": ["blood_sugar_control"],
    "is_primary": true,
    "target_values": {"hba1c": 7.0}
  }'

# 4. Update health context
curl -X PUT "http://localhost:8000/api/v1/health/context" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "age_group": "adult",
    "activity_level": "moderate",
    "weight_kg": 70.0,
    "height_cm": 175.0,
    "preferred_language": "fr"
  }'
```

### 3. Food Product Analysis
```bash
# 5. Analyze a food product
curl -X POST "http://localhost:8000/api/v1/analysis/scan-food" \
  -H "Authorization: Bearer <access_token>" \
  -F "image=@./food_image.jpg" \
  -F "barcode=3017620422003" \
  -F "integration_level=expert_analysis"
```

### 4. View Results & Chat
```bash
# 6. Get analysis history
curl -X GET "http://localhost:8000/api/v1/analysis/history" \
  -H "Authorization: Bearer <access_token>"

# 7. Ask AI about the product
curl -X POST "http://localhost:8000/api/v1/chat/message" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Est-ce que je peux manger ce produit avec mon diabète ?",
    "context": {"analysis_id": "<analysis_id>"}
  }'
```

## 📊 Response Examples

### Health Analysis Response
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "product": {
    "name": "Nutella",
    "brand": "Ferrero",
    "categories": ["spreads", "sweet-spreads"]
  },
  "nutrition": {
    "energy_kcal": 539,
    "fat": 30.9,
    "saturated_fat": 10.6,
    "carbohydrates": 57.5,
    "sugars": 56.3,
    "fiber": 1.9,
    "protein": 6.3,
    "salt": 0.107
  },
  "health_analysis": {
    "diabetes": {
      "score": 2,
      "risk_level": "high",
      "key_concerns": [
        "Très haute teneur en sucres (56.3g/100g)",
        "Index glycémique élevé",
        "Portion recommandée très limitée"
      ],
      "recommendations": [
        "❌ À éviter : Teneur en sucres excessive pour le diabète",
        "🔄 Alternative : Purée d'amandes sans sucre ajouté",
        "⚠️ Si consommation : Maximum 1 cuillère à café (5g)"
      ],
      "portion_guidance": {
        "max_portion": "5g",
        "frequency": "exceptionnel",
        "timing": "après activité physique"
      }
    }
  },
  "overall_score": 2,
  "recommendation": "avoid",
  "ai_insights": "En tant que nutritionniste spécialisé, je dois vous déconseiller fortement ce produit...",
  "created_at": "2025-09-17T10:30:00Z"
}
```

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
