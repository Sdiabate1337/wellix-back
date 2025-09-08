"""
Chat session and message models for AI conversation tracking.
"""

from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.database import Base


class ChatSession(Base):
    """Chat session for tracking AI conversations with context."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Session metadata
    session_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Context tracking
    related_analyses = Column(ARRAY(UUID), default=list)  # FoodAnalysis IDs
    context_summary = Column(Text, nullable=True)
    
    # AI model configuration
    ai_model = Column(String(100), default="gpt-4")
    system_prompt = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class ChatMessage(Base):
    """Individual chat messages with AI responses."""
    
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Message metadata
    message_type = Column(String(50), default="text")  # text, analysis_reference, recommendation
    referenced_analysis_id = Column(UUID(as_uuid=True), nullable=True)
    
    # AI processing
    ai_model = Column(String(100), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    # Additional context
    context_data = Column(JSONB, default=dict)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})>"
