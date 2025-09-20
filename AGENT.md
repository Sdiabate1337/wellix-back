# 🧠 WELLIX SaaS - Architecture LangGraph avec Système de Tokens

## 📋 Vision Globale

**Objectif :** SaaS d'analyse nutritionnelle personnalisée avec IA, monétisé par système de tokens.

**Inputs Utilisateur :**
- 📷 Scan image fiche nutritionnelle (OCR)
- 📝 Saisie manuelle données nutritionnelles

**Outputs :**
- 🎯 Score personnalisé par condition médicale
- 💡 10 alternatives produits plus saines
- 💬 Contexte chat pour discussions approfondies

**Monétisation :** Tokens avec plans freemium (Free: 20 tokens/mois → Premium: 500 tokens/mois)

---

## 🏗️ Architecture Technique

### 🔄 Workflow LangGraph Principal

```mermaid
graph TD
    A[📱 Input: Image/Manual] → B[🔍 Extract Nutrition Data]
    B → C[👤 Load User Health Profile]
    C → D[🎯 Validate Tokens Available]
    D →|❌ No Tokens| E[💳 Upgrade Prompt]
    D →|✅ Has Tokens| F[🧠 LangGraph Analysis]
    F → G[🔬 Expert Medical Prompts]
    G → H[🥗 Generate Alternatives]
    H → I[💬 Create Chat Context]
    I → J[💰 Deduct Tokens]
    J → K[📱 Return Results]
```

### 🎯 États du Workflow

```python
@dataclass
class NutritionAnalysisState:
    # Input Data
    image_data: Optional[bytes]
    manual_nutrition: Optional[Dict]
    user_id: UUID
    
    # User Context  
    user_profile: UserHealthContext
    health_conditions: List[HealthProfile]
    available_tokens: int
    
    # Processing Results
    nutrition_data: Optional[Dict]
    ocr_confidence: Optional[float]
    
    # AI Analysis Results
    condition_analyses: Dict[str, Any]  # Par condition médicale
    overall_score: int
    safety_level: str
    
    # Product Alternatives
    alternatives: List[ProductAlternative]
    
    # Chat Integration
    chat_context: Dict[str, Any]
    
    # System Metadata
    tokens_consumed: int
    processing_steps: List[str]
    errors: List[str]
    total_processing_time_ms: int
```

---

## 🧠 Système de Prompts Experts

### 📝 Template Principal par Condition

```python
# Optimisé : Un prompt combiné pour économiser tokens
COMBINED_EXPERT_PROMPT = """
Tu es un expert nutritionniste spécialisé en {conditions_list}.

DONNÉES NUTRITIONNELLES:
{nutrition_data}

PROFIL PATIENT:
{user_health_profile}

CONDITIONS À ANALYSER:
{health_conditions_details}

TÂCHES REQUISES:
1. Score pour chaque condition (0-100)
2. Justification médicale par condition
3. Niveau de sécurité global (safe/caution/warning/danger)
4. Recommandations spécifiques par condition
5. Timing optimal de consommation
6. Interactions potentielles avec médicaments

CONTRAINTES:
- Réponse en JSON structuré uniquement
- Disclaimer: "Consultation médicale recommandée"
- Maximum 500 tokens de réponse

FORMAT ATTENDU:
{
  "conditions_analysis": {
    "diabetes": {"score": 75, "reasoning": "...", "safety": "caution"},
    "hypertension": {"score": 40, "reasoning": "...", "safety": "warning"}
  },
  "overall_assessment": {
    "global_score": 58,
    "safety_level": "warning", 
    "key_concerns": ["high_sodium", "added_sugars"],
    "portion_recommendation": "50g maximum"
  },
  "medical_disclaimer": "Cette analyse est informative uniquement..."
}
"""
```

### 🎯 Stratégies par Condition

```python
# app/strategies/expert_prompts/
class DiabetesExpertStrategy:
    focus_nutrients = ["carbs", "sugar", "fiber", "glycemic_index"]
    critical_thresholds = {"sugar_per_100g": 15, "carbs_per_100g": 60}
    
class HypertensionExpertStrategy:
    focus_nutrients = ["sodium", "potassium", "saturated_fat"]
    critical_thresholds = {"sodium_per_100g": 600, "saturated_fat": 5}
    
class GeneralHealthStrategy:
    focus_nutrients = ["calories", "protein", "vitamins", "additives"]
    processing_level_impact = True
```

---

## 💰 Système de Monétisation

### 🎫 Coûts par Token

```python
TOKEN_COSTS = {
    # Analyses
    "basic_analysis": 1,           # Score simple sans IA
    "expert_analysis": 5,          # Analyse complète avec prompts experts
    "multi_condition_analysis": 7, # Plusieurs conditions médicales
    
    # Services additionnels
    "alternatives_generation": 3,   # 10 alternatives personnalisées
    "chat_interaction": 1,         # Par message de chat
    "image_ocr_processing": 2,     # OCR d'image nutritionnelle
    "detailed_recommendations": 2,  # Recommandations détaillées
    
    # Features premium
    "meal_planning": 10,           # Planning repas personnalisé
    "progress_tracking": 5,        # Suivi évolution santé
}

SUBSCRIPTION_PLANS = {
    "free": {
        "monthly_tokens": 20,
        "price_eur": 0,
        "features": ["basic_analysis", "chat_interaction"]
    },
    "basic": {
        "monthly_tokens": 100, 
        "price_eur": 9.99,
        "features": ["expert_analysis", "alternatives_generation", "image_ocr"]
    },
    "premium": {
        "monthly_tokens": 500,
        "price_eur": 29.99,
        "features": ["all_features", "priority_support", "advanced_insights"]
    },
    "enterprise": {
        "monthly_tokens": 2000,
        "price_eur": 99.99,
        "features": ["unlimited_analysis", "custom_prompts", "api_access"]
    }
}
```

### 💳 Économie du Modèle

```python
# Calcul économique optimal
ESTIMATED_LLM_COSTS = {
    "gpt-4o-mini": 0.02,  # Par analyse expert complète
    "gpt-4o": 0.08,       # Analyse premium detailed
    "claude-3": 0.05,     # Alternative équilibrée
}

# Marge cible : 70%
# Prix utilisateur : 5 tokens = ~0.17€
# Coût réel LLM : ~0.05€
# Marge brute : 0.12€ (70%)
```

---

## 🗄️ Modèles de Données

### 🎫 Système de Tokens

```python
class UserTokenBalance(Base):
    __tablename__ = "user_token_balances"
    
    user_id = Column(UUID, ForeignKey("users.id"), primary_key=True)
    plan_type = Column(String(20), nullable=False)  # free, basic, premium
    
    # Quotas mensuels
    monthly_token_quota = Column(Integer, nullable=False)
    
    # Usage actuel
    tokens_used_this_month = Column(Integer, default=0)
    tokens_remaining = Column(Integer, nullable=False)
    
    # Dates importantes
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    last_reset_date = Column(DateTime, nullable=False)
    
    # Métadonnées
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class TokenTransaction(Base):
    __tablename__ = "token_transactions"
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID, ForeignKey("users.id"), nullable=False)
    
    # Transaction details
    amount = Column(Integer, nullable=False)  # Négatif = dépense, Positif = crédit
    transaction_type = Column(String(50), nullable=False)
    feature_used = Column(String(100), nullable=True)
    
    # Context
    analysis_id = Column(UUID, ForeignKey("food_analyses.id"), nullable=True)
    chat_session_id = Column(UUID, nullable=True)
    
    # Métadonnées
    created_at = Column(DateTime, server_default=func.now())
    processing_time_ms = Column(Integer, nullable=True)
```

### 🧠 Prompts et Configuration

```python
class ExpertPromptTemplate(Base):
    __tablename__ = "expert_prompt_templates"
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    condition_type = Column(String(50), nullable=False)  # diabetes, hypertension
    prompt_version = Column(String(20), nullable=False)
    
    # Templates de prompts
    system_prompt = Column(Text, nullable=False)
    analysis_template = Column(Text, nullable=False)
    alternatives_template = Column(Text, nullable=True)
    
    # Configuration
    max_tokens = Column(Integer, default=500)
    temperature = Column(Float, default=0.1)
    is_active = Column(Boolean, default=True)
    
    # Métriques qualité
    success_rate = Column(Float, default=1.0)
    average_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
```

---

## 🔧 Nœuds LangGraph Détaillés

### 1️⃣ Nœud de Validation Tokens

```python
async def validate_tokens_node(state: NutritionAnalysisState) -> NutritionAnalysisState:
    """Vérifie la disponibilité des tokens avant traitement."""
    
    required_tokens = calculate_required_tokens(state)
    user_balance = await token_repository.get_user_balance(state.user_id)
    
    if user_balance.tokens_remaining < required_tokens:
        state.errors.append("insufficient_tokens")
        state.upgrade_suggestion = suggest_plan_upgrade(user_balance, required_tokens)
        return state
    
    # Réserver les tokens (transaction pending)
    await token_repository.reserve_tokens(state.user_id, required_tokens)
    state.tokens_reserved = required_tokens
    
    return state
```

### 2️⃣ Nœud d'Analyse Expert

```python
async def expert_analysis_node(state: NutritionAnalysisState) -> NutritionAnalysisState:
    """Analyse par IA avec prompts experts par condition médicale."""
    
    if not state.nutrition_data or state.errors:
        return state
    
    analyses = {}
    
    # Optimisation: Grouper conditions similaires dans un prompt
    condition_groups = group_conditions_by_similarity(state.health_conditions)
    
    for group in condition_groups:
        try:
            # Prompt combiné pour économiser tokens
            prompt = await expert_prompt_factory.create_combined_prompt(
                conditions=group,
                nutrition_data=state.nutrition_data,
                user_profile=state.user_profile
            )
            
            # Appel LLM avec retry automatique
            analysis_result = await llm_service.analyze_with_retry(
                prompt=prompt,
                max_retries=2,
                fallback_to_algorithmic=True
            )
            
            # Parser et valider résultats
            parsed_results = await analysis_parser.parse_expert_response(
                analysis_result, group
            )
            
            analyses.update(parsed_results)
            
        except Exception as e:
            # Fallback algorithmique en cas d'échec LLM
            fallback_results = await algorithmic_analyzer.analyze_conditions(
                group, state.nutrition_data
            )
            analyses.update(fallback_results)
            
            state.errors.append(f"llm_fallback_used: {str(e)}")
    
    state.condition_analyses = analyses
    state.overall_score = calculate_weighted_score(analyses)
    state.safety_level = determine_overall_safety(analyses)
    
    return state
```

### 3️⃣ Nœud de Génération d'Alternatives

```python
async def generate_alternatives_node(state: NutritionAnalysisState) -> NutritionAnalysisState:
    """Génère 10 alternatives produits plus saines."""
    
    if not state.condition_analyses or state.errors:
        return state
    
    try:
        # Analyse des points faibles du produit actuel
        weak_points = extract_nutritional_weaknesses(state.condition_analyses)
        
        # Recherche dans base de données produits
        similar_products = await product_repository.find_similar_products(
            category=state.nutrition_data.get("category"),
            exclude_barcode=state.nutrition_data.get("barcode"),
            limit=50
        )
        
        # Scoring des alternatives
        scored_alternatives = []
        for product in similar_products:
            alternative_score = await score_alternative_product(
                product=product,
                user_conditions=state.health_conditions,
                reference_weak_points=weak_points
            )
            scored_alternatives.append((product, alternative_score))
        
        # Top 10 alternatives
        top_alternatives = sorted(scored_alternatives, 
                                key=lambda x: x[1], reverse=True)[:10]
        
        # Enrichir avec justifications IA
        enriched_alternatives = []
        for product, score in top_alternatives:
            justification = await llm_service.generate_alternative_justification(
                original=state.nutrition_data,
                alternative=product,
                user_conditions=state.health_conditions,
                improvement_score=score
            )
            
            enriched_alternatives.append({
                "product": product,
                "improvement_score": score,
                "justification": justification,
                "key_improvements": calculate_improvements(
                    state.nutrition_data, product
                )
            })
        
        state.alternatives = enriched_alternatives
        
    except Exception as e:
        state.errors.append(f"alternatives_generation_failed: {str(e)}")
        state.alternatives = []
    
    return state
```

---

## 🛡️ Sécurité et Conformité

### ⚖️ Disclaimers Médicaux Obligatoires

```python
MEDICAL_DISCLAIMERS = {
    "analysis": """
    ⚠️ AVERTISSEMENT MÉDICAL
    Cette analyse nutritionnelle est fournie à titre informatif uniquement. 
    Elle ne remplace pas l'avis d'un professionnel de santé qualifié.
    Consultez votre médecin avant de modifier votre alimentation.
    """,
    
    "diabetes": """
    🩺 ATTENTION DIABÈTE
    Les recommandations pour le diabète sont générales. 
    Votre glycémie peut réagir différemment. 
    Surveillez votre taux et consultez votre endocrinologue.
    """,
    
    "hypertension": """
    💔 ATTENTION HYPERTENSION
    L'impact sur la tension artérielle varie selon les individus.
    Continuez votre traitement prescrit et consultez votre cardiologue.
    """
}
```

### 🔒 Rate Limiting et Protection

```python
# Middleware de protection
@rate_limit(requests_per_minute=30, per_user=True)
@token_required(min_tokens=1)
@validate_input_data()
async def nutrition_analysis_endpoint(request: AnalysisRequest):
    """Endpoint protégé pour analyse nutritionnelle."""
    pass

# Protection contre abus
class AbuseProtection:
    max_daily_analyses = 100
    max_images_per_hour = 20
    suspicious_pattern_detection = True
    auto_ban_threshold = 500  # analyses/jour
```

---

## 📊 Métriques et Analytics

### 📈 KPIs Business Essentiels

```python
BUSINESS_METRICS = {
    # Monétisation
    "revenue_per_user": "Revenus moyens par utilisateur",
    "token_conversion_rate": "% users free → payant", 
    "churn_rate": "Taux de désabonnement mensuel",
    "lifetime_value": "Valeur vie client",
    
    # Usage
    "daily_active_users": "Utilisateurs actifs quotidiens",
    "analyses_per_user": "Analyses moyennes par utilisateur",
    "chat_engagement": "Messages chat par session",
    "feature_adoption": "Adoption nouvelles fonctionnalités",
    
    # Qualité
    "analysis_accuracy": "Précision analyses vs feedback",
    "user_satisfaction": "Score satisfaction (1-5)",
    "support_tickets": "Tickets support par mois",
    "system_uptime": "Disponibilité système"
}
```

### 🔍 Tracking Automatique

```python
# Observer pattern pour métriques
class AnalyticsObserver:
    async def on_analysis_completed(self, analysis_result):
        await metrics_service.track_analysis_completion(
            user_id=analysis_result.user_id,
            processing_time=analysis_result.processing_time_ms,
            tokens_consumed=analysis_result.tokens_consumed,
            conditions_analyzed=len(analysis_result.condition_analyses),
            alternatives_generated=len(analysis_result.alternatives)
        )
    
    async def on_token_transaction(self, transaction):
        await metrics_service.track_token_usage(
            user_id=transaction.user_id,
            amount=transaction.amount,
            feature=transaction.feature_used,
            plan_type=transaction.user_plan
        )
```

---

## 🚀 Plan d'Implémentation Optimisé

### 📅 Phase 1: Foundation (2-3 semaines)
- ✅ Système de tokens avec quotas
- ✅ Repositories (User, Nutrition, Tokens)
- ✅ Middleware de validation tokens
- ✅ Modèles de données essentiels

### 📅 Phase 2: Core Analysis (2-3 semaines)  
- ✅ Workflow LangGraph simplifié (5 nœuds principaux)
- ✅ Prompts experts par condition
- ✅ Service OCR intégré
- ✅ Analyses algorithmiques de fallback

### 📅 Phase 3: Intelligence (2 semaines)
- ✅ Génération alternatives produits
- ✅ Chat contextuel intelligent
- ✅ Optimisations économiques LLM
- ✅ Métriques et analytics

### 📅 Phase 4: Scale & Polish (1-2 semaines)
- ✅ Rate limiting avancé
- ✅ Cache intelligent
- ✅ Monitoring et alertes
- ✅ Tests de charge

---

## ✅ Validation du Plan

### ✅ **Avantages Confirmés**
- **Business Model Validé :** Tokens = revenus prévisibles et scalables
- **Architecture Robuste :** LangGraph + Patterns = maintenabilité
- **UX Optimale :** Double input + chat = engagement maximum
- **Économie Équilibrée :** Marge 70% avec fallbacks algorithmiques

### ⚠️ **Risques Contrôlés**
- **Complexité :** Phases progressives pour réduire le risque
- **Coûts LLM :** Prompts optimisés + cache + fallbacks
- **Légal :** Disclaimers médicaux obligatoires
- **Performance :** Rate limiting + monitoring

### 🎯 **Facteurs de Succès**
1. **MVP Rapide :** Version simplifiée en 4-6 semaines
2. **Feedback Loop :** Métriques en temps réel pour optimisation
3. **Extensibilité :** Architecture prête pour nouvelles conditions médicales
4. **Monétisation :** Modèle freemium avec conversion naturelle

---

## 📝 Prochaines Actions

1. **[PRIORITÉ 1]** Implémenter système de tokens et repositories
2. **[PRIORITÉ 2]** Créer workflow LangGraph de base (3 nœuds)
3. **[PRIORITÉ 3]** Développer prompts experts diabète + hypertension
4. **[PRIORITÉ 4]** Tester avec produit exemple (Nutella)
5. **[PRIORITÉ 5]** Itérer basé sur résultats tests

---

**🎯 Cette architecture est validée et prête pour implémentation progressive !**