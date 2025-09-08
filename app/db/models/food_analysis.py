"""
Food analysis database models for storing OCR results and health analysis.
"""

from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.database import Base


class FoodAnalysis(Base):
    """Complete food analysis result with nutrition data and health scoring."""
    
    __tablename__ = "food_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Product identification
    product_name = Column(String(255), nullable=False)
    brand = Column(String(255), nullable=True)
    barcode = Column(String(50), nullable=True, index=True)
    
    # Nutrition data (per serving)
    serving_size = Column(String(100), nullable=False)
    calories = Column(Float, nullable=False)
    protein = Column(Float, nullable=False)
    carbohydrates = Column(Float, nullable=False)
    total_fat = Column(Float, nullable=False)
    saturated_fat = Column(Float, nullable=True)
    trans_fat = Column(Float, nullable=True)
    fiber = Column(Float, nullable=True)
    sugar = Column(Float, nullable=True)
    added_sugar = Column(Float, nullable=True)
    sodium = Column(Float, nullable=True)
    potassium = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    
    # Additional nutrition data
    micronutrients = Column(JSONB, default=dict)  # vitamins, minerals
    ingredients = Column(ARRAY(String), default=list)
    allergens = Column(ARRAY(String), default=list)
    additives = Column(ARRAY(String), default=list)
    
    # Analysis results
    overall_score = Column(Integer, nullable=False)
    safety_score = Column(Integer, nullable=False)
    profile_scores = Column(JSONB, default=dict)  # condition-specific scores
    
    # Insights
    safety_alerts = Column(ARRAY(String), default=list)
    recommendations = Column(ARRAY(String), default=list)
    warnings = Column(JSONB, default=list)
    
    # Additional analysis
    glycemic_impact = Column(Integer, nullable=True)
    processing_level = Column(String(50), nullable=True)
    ingredient_quality_score = Column(Integer, nullable=True)
    
    # Data source and confidence
    data_source = Column(String(50), default="manual")
    confidence_score = Column(Float, nullable=True)
    ocr_text = Column(Text, nullable=True)  # Raw OCR output
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    ai_model_used = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="food_analyses")
    recommendations = relationship("ProductRecommendation", back_populates="analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<FoodAnalysis(id={self.id}, product={self.product_name}, score={self.overall_score})>"


class ProductRecommendation(Base):
    """Product recommendations based on health analysis."""
    
    __tablename__ = "product_recommendations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("food_analyses.id"), nullable=False, index=True)
    
    # Recommended product
    product_name = Column(String(255), nullable=False)
    brand = Column(String(255), nullable=False)
    barcode = Column(String(50), nullable=True)
    
    # Recommendation scoring
    recommendation_score = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    health_benefits = Column(ARRAY(String), default=list)
    
    # Availability and pricing
    availability = Column(String(100), nullable=True)
    price_comparison = Column(String(100), nullable=True)
    
    # Nutrition improvements
    nutrition_improvements = Column(JSONB, default=dict)
    
    # Vector similarity data
    similarity_score = Column(Float, nullable=True)
    vector_id = Column(String(100), nullable=True)  # Pinecone vector ID
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    analysis = relationship("FoodAnalysis", back_populates="recommendations")
    
    def __repr__(self):
        return f"<ProductRecommendation(id={self.id}, product={self.product_name}, score={self.recommendation_score})>"


class OCRResult(Base):
    """Raw OCR results for audit and improvement."""
    
    __tablename__ = "ocr_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("food_analyses.id"), nullable=True, index=True)
    
    # Image data
    image_url = Column(String(500), nullable=True)
    image_hash = Column(String(64), nullable=True, index=True)
    
    # OCR processing
    ocr_service = Column(String(50), nullable=False)  # google_vision, tesseract, etc.
    raw_text = Column(Text, nullable=False)
    structured_data = Column(JSONB, default=dict)
    confidence_scores = Column(JSONB, default=dict)
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<OCRResult(id={self.id}, service={self.ocr_service})>"
