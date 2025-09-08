"""
Health profile database models for multi-condition analysis.
"""

from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.database import Base


class HealthProfile(Base):
    """Health profile model for storing user health conditions and preferences."""
    
    __tablename__ = "health_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Profile type and severity
    profile_type = Column(String(50), nullable=False)  # diabetes, hypertension, etc.
    severity = Column(String(20), nullable=False)  # mild, moderate, severe
    is_primary = Column(Boolean, default=False)
    
    # Health data
    restrictions = Column(ARRAY(String), default=list)
    goals = Column(ARRAY(String), default=list)
    medications = Column(ARRAY(String), default=list)
    target_values = Column(JSONB, default=dict)
    
    # Additional context
    notes = Column(Text, nullable=True)
    diagnosis_date = Column(DateTime(timezone=True), nullable=True)
    last_checkup = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="health_profiles")
    
    def __repr__(self):
        return f"<HealthProfile(id={self.id}, type={self.profile_type}, severity={self.severity})>"


class UserHealthContext(Base):
    """Complete user health context and preferences."""
    
    __tablename__ = "user_health_contexts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    # Basic health info
    age_group = Column(String(20), nullable=False)
    activity_level = Column(String(30), nullable=False)
    weight_goals = Column(String(50), nullable=True)
    
    # Physical measurements
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    
    # Allergies and preferences
    allergies = Column(ARRAY(String), default=list)
    dietary_preferences = Column(ARRAY(String), default=list)
    
    # Analysis preferences
    analysis_depth = Column(String(20), default="standard")
    preferred_language = Column(String(10), default="en")
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<UserHealthContext(id={self.id}, user_id={self.user_id})>"
