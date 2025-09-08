"""
WebSocket endpoints for real-time communication and updates.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.websockets import WebSocketState
import json
import asyncio
import structlog
from datetime import datetime

from app.core.dependencies import get_current_user_from_token
from app.db.database import get_async_session
from app.db.models.user import User
from app.db.models.chat import ChatSession, ChatMessage
from app.services.ai.chat_service import chat_service
from app.cache.cache_manager import cache_manager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = structlog.get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_sessions: Dict[str, str] = {}  # websocket_id -> user_id
    
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str):
        """Accept WebSocket connection and register user."""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        self.user_sessions[id(websocket)] = user_id
        
        logger.info(f"WebSocket connected for user {user_id}, session {session_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        websocket_id = id(websocket)
        user_id = self.user_sessions.get(websocket_id)
        
        if user_id and user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            except ValueError:
                pass
        
        if websocket_id in self.user_sessions:
            del self.user_sessions[websocket_id]
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user's connections."""
        if user_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[user_id]:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(message))
                    else:
                        disconnected.append(websocket)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast_to_session(self, message: dict, session_id: str):
        """Broadcast message to all users in a session (future feature)."""
        # For now, this is the same as personal message
        # In future, could support shared sessions
        pass


manager = ConnectionManager()


@router.websocket("/chat/{session_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = None
):
    """WebSocket endpoint for real-time chat communication."""
    user = None
    
    try:
        # Authenticate user from token
        if token:
            try:
                user = await get_current_user_from_token(token)
            except HTTPException:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Connect to WebSocket
        await manager.connect(websocket, str(user.id), session_id)
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                await handle_websocket_message(
                    websocket, user, session_id, message_data
                )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception as e:
                logger.error(f"WebSocket error for user {user.id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat()
                }))
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        manager.disconnect(websocket)


async def handle_websocket_message(
    websocket: WebSocket,
    user: User,
    session_id: str,
    message_data: dict
):
    """Handle incoming WebSocket messages."""
    message_type = message_data.get("type")
    
    if message_type == "chat_message":
        await handle_chat_message(websocket, user, session_id, message_data)
    
    elif message_type == "typing":
        await handle_typing_indicator(websocket, user, session_id, message_data)
    
    elif message_type == "ping":
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    else:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        }))


async def handle_chat_message(
    websocket: WebSocket,
    user: User,
    session_id: str,
    message_data: dict
):
    """Handle chat message through WebSocket."""
    try:
        message_content = message_data.get("message", "").strip()
        if not message_content:
            return
        
        # Get database session
        async with get_async_session() as db:
            # Verify chat session exists and belongs to user
            result = await db.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user.id
                )
            )
            chat_session = result.scalar_one_or_none()
            
            if not chat_session:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Chat session not found",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                return
            
            # Save user message
            user_message = ChatMessage(
                session_id=session_id,
                role="user",
                content=message_content,
                metadata={"source": "websocket"}
            )
            db.add(user_message)
            await db.commit()
            await db.refresh(user_message)
            
            # Send confirmation of user message
            await websocket.send_text(json.dumps({
                "type": "message_received",
                "message_id": str(user_message.id),
                "timestamp": user_message.created_at.isoformat()
            }))
            
            # Generate AI response with streaming
            try:
                ai_response_id = None
                full_response = ""
                
                async for chunk in chat_service.stream_chat_response(
                    message_content, str(user.id), session_id
                ):
                    if chunk.get("type") == "content":
                        content = chunk.get("content", "")
                        full_response += content
                        
                        # Send streaming chunk
                        await websocket.send_text(json.dumps({
                            "type": "ai_response_chunk",
                            "content": content,
                            "message_id": ai_response_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    
                    elif chunk.get("type") == "message_start":
                        ai_response_id = chunk.get("message_id")
                        
                        # Send response start
                        await websocket.send_text(json.dumps({
                            "type": "ai_response_start",
                            "message_id": ai_response_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                
                # Send response complete
                await websocket.send_text(json.dumps({
                    "type": "ai_response_complete",
                    "message_id": ai_response_id,
                    "full_content": full_response,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
            except Exception as e:
                logger.error(f"Error generating AI response: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to generate AI response",
                    "timestamp": datetime.utcnow().isoformat()
                }))
    
    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Failed to process message",
            "timestamp": datetime.utcnow().isoformat()
        }))


async def handle_typing_indicator(
    websocket: WebSocket,
    user: User,
    session_id: str,
    message_data: dict
):
    """Handle typing indicator."""
    is_typing = message_data.get("is_typing", False)
    
    # For now, just acknowledge the typing indicator
    # In a multi-user chat, this would broadcast to other users
    await websocket.send_text(json.dumps({
        "type": "typing_acknowledged",
        "is_typing": is_typing,
        "timestamp": datetime.utcnow().isoformat()
    }))


@router.websocket("/analysis/{analysis_id}/updates")
async def websocket_analysis_updates(
    websocket: WebSocket,
    analysis_id: str,
    token: Optional[str] = None
):
    """WebSocket endpoint for real-time analysis updates."""
    user = None
    
    try:
        # Authenticate user
        if token:
            try:
                user = await get_current_user_from_token(token)
            except HTTPException:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await websocket.accept()
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Monitor analysis progress (placeholder for future implementation)
        # This would integrate with the workflow system to send real-time updates
        
        while True:
            try:
                # Keep connection alive and handle any client messages
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                if message_data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Analysis WebSocket error: {e}")
                break
    
    except Exception as e:
        logger.error(f"Analysis WebSocket connection error: {e}")
    
    finally:
        logger.info(f"Analysis WebSocket disconnected for user {user.id if user else 'unknown'}")


# Add WebSocket connection manager to dependency injection
async def get_connection_manager() -> ConnectionManager:
    """Get WebSocket connection manager."""
    return manager
