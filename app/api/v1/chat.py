"""
AI chat API endpoints for context-aware food analysis conversations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import structlog
import json
from datetime import datetime
import uuid

from app.core.dependencies import get_current_active_user, get_optional_user
from app.db.database import get_async_session
from app.db.models.user import User
from app.db.models.chat import ChatSession, ChatMessage
from app.services.ai.chat_service import chat_service
from app.cache.cache_manager import cache_manager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

router = APIRouter()


class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    analysis_id: Optional[str] = None
    provider: str = "openai"
    model: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    timestamp: str
    usage: Optional[Dict[str, Any]] = None
    suggested_questions: Optional[List[str]] = None


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Send a message to the AI chat system with full context awareness.
    """
    try:
        # Get or create chat session
        session = await _get_or_create_session(request.session_id, current_user.id, db)
        
        # Get user health context
        from app.api.v1.analysis import _get_user_health_context
        user_context = await _get_user_health_context(current_user.id, db)
        
        # Get analysis context if provided
        analysis_context = None
        if request.analysis_id:
            analysis_context = await cache_manager.get_food_analysis(request.analysis_id)
            if analysis_context:
                # Add analysis to session context
                if request.analysis_id not in session.related_analyses:
                    session.related_analyses.append(request.analysis_id)
                    await db.commit()
        
        # Get chat history
        chat_history = await _get_chat_history(session.id, db)
        
        # Save user message
        user_message = ChatMessage(
            session_id=session.id,
            role="user",
            content=request.message,
            referenced_analysis_id=request.analysis_id
        )
        db.add(user_message)
        await db.commit()
        
        # Get AI response
        if request.stream:
            return StreamingResponse(
                _stream_chat_response(
                    request, session, user_context, analysis_context, chat_history, db
                ),
                media_type="text/plain"
            )
        else:
            ai_response = await chat_service.chat_with_context(
                user_message=request.message,
                user_context=user_context,
                chat_history=chat_history,
                analysis_context=analysis_context,
                provider=request.provider,
                model=request.model,
                stream=False
            )
            
            # Save AI response
            ai_message = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=ai_response["response"],
                ai_model=request.model or f"{request.provider}-default",
                processing_time_ms=ai_response.get("processing_time_ms"),
                token_count=ai_response.get("usage", {}).get("total_tokens")
            )
            db.add(ai_message)
            await db.commit()
            
            # Generate suggested questions
            suggested_questions = []
            if analysis_context:
                suggested_questions = await chat_service.suggest_follow_up_questions(
                    analysis_context, user_context
                )
            
            return ChatResponse(
                response=ai_response["response"],
                session_id=str(session.id),
                message_id=str(ai_message.id),
                timestamp=datetime.utcnow().isoformat(),
                usage=ai_response.get("usage"),
                suggested_questions=suggested_questions
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message error for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@router.get("/sessions")
async def get_chat_sessions(
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get user's chat sessions."""
    try:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == current_user.id)
            .order_by(desc(ChatSession.updated_at))
            .limit(limit)
        )
        sessions = result.scalars().all()
        
        session_list = []
        for session in sessions:
            # Get last message for preview
            last_message_result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session.id)
                .order_by(desc(ChatMessage.created_at))
                .limit(1)
            )
            last_message = last_message_result.scalar_one_or_none()
            
            session_list.append({
                "session_id": str(session.id),
                "session_name": session.session_name or "Chat Session",
                "is_active": session.is_active,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "last_message": last_message.content[:100] + "..." if last_message and len(last_message.content) > 100 else last_message.content if last_message else None,
                "related_analyses": session.related_analyses
            })
        
        return JSONResponse(content={"sessions": session_list})
        
    except Exception as e:
        logger.error(f"Error retrieving chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat sessions"
        )


@router.get("/sessions/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get messages from a chat session."""
    try:
        # Verify session ownership
        session_result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == current_user.id
            )
        )
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Get messages
        messages_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .offset(offset)
        )
        messages = messages_result.scalars().all()
        
        message_list = []
        for message in messages:
            message_list.append({
                "message_id": str(message.id),
                "role": message.role,
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "referenced_analysis_id": str(message.referenced_analysis_id) if message.referenced_analysis_id else None,
                "ai_model": message.ai_model
            })
        
        return JSONResponse(content={
            "session_id": session_id,
            "messages": message_list,
            "total": len(message_list)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat messages"
        )


@router.websocket("/ws/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    try:
        # Note: In production, you'd want to authenticate the WebSocket connection
        # For now, we'll assume the session_id is valid
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            analysis_id = message_data.get("analysis_id")
            provider = message_data.get("provider", "openai")
            
            if not user_message:
                await websocket.send_text(json.dumps({
                    "error": "Empty message"
                }))
                continue
            
            # Process message (simplified for WebSocket)
            try:
                # Get basic context (you'd want to implement proper auth here)
                from app.models.health import UserHealthContext, AgeGroup, ActivityLevel
                user_context = UserHealthContext(
                    user_id="websocket_user",  # Replace with actual user ID
                    age_group=AgeGroup.ADULT,
                    activity_level=ActivityLevel.MODERATELY_ACTIVE
                )
                
                # Get analysis context if provided
                analysis_context = None
                if analysis_id:
                    analysis_context = await cache_manager.get_food_analysis(analysis_id)
                
                # Stream response
                async for chunk in chat_service.chat_with_context(
                    user_message=user_message,
                    user_context=user_context,
                    chat_history=[],
                    analysis_context=analysis_context,
                    provider=provider,
                    stream=True
                ):
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "content": chunk.get("chunk", ""),
                        "timestamp": chunk.get("timestamp")
                    }))
                
                # Send completion signal
                await websocket.send_text(json.dumps({
                    "type": "complete"
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


async def _get_or_create_session(
    session_id: Optional[str],
    user_id: str,
    db: AsyncSession
) -> ChatSession:
    """Get existing session or create new one."""
    if session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            )
        )
        session = result.scalar_one_or_none()
        if session:
            return session
    
    # Create new session
    session = ChatSession(
        user_id=user_id,
        session_name=f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        ai_model="gpt-4"
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return session


async def _get_chat_history(session_id: str, db: AsyncSession) -> List[Dict[str, str]]:
    """Get recent chat history for context."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(desc(ChatMessage.created_at))
        .limit(10)  # Last 10 messages for context
    )
    messages = result.scalars().all()
    
    # Reverse to get chronological order
    history = []
    for message in reversed(messages):
        history.append({
            "role": message.role,
            "content": message.content
        })
    
    return history


async def _stream_chat_response(
    request: ChatMessageRequest,
    session: ChatSession,
    user_context,
    analysis_context,
    chat_history: List[Dict[str, str]],
    db: AsyncSession
):
    """Stream chat response for real-time updates."""
    full_response = ""
    
    try:
        async for chunk in chat_service.chat_with_context(
            user_message=request.message,
            user_context=user_context,
            chat_history=chat_history,
            analysis_context=analysis_context,
            provider=request.provider,
            model=request.model,
            stream=True
        ):
            chunk_text = chunk.get("chunk", "")
            full_response += chunk_text
            
            # Send chunk to client
            yield f"data: {json.dumps({'chunk': chunk_text, 'timestamp': chunk.get('timestamp')})}\n\n"
        
        # Save complete AI response
        ai_message = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=full_response,
            ai_model=request.model or f"{request.provider}-default"
        )
        db.add(ai_message)
        await db.commit()
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete', 'message_id': str(ai_message.id)})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
