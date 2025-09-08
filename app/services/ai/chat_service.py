"""
AI chat service for context-aware food analysis conversations.
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
import structlog
from datetime import datetime

from app.services.ai.ai_service import ai_service_manager
from app.cache.cache_manager import cache_manager
from app.models.health import UserHealthContext

logger = structlog.get_logger(__name__)


class ChatService:
    """AI chat service with food analysis context integration."""
    
    def __init__(self):
        self.ai_manager = ai_service_manager
        self.default_system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for food analysis chat."""
        return """You are Wellix AI, an expert nutritionist and health advisor specializing in personalized food analysis. You help users understand their food choices based on their specific health conditions and dietary needs.

Your capabilities include:
- Analyzing nutrition labels and food products
- Providing personalized recommendations based on health conditions (diabetes, hypertension, heart disease, etc.)
- Explaining nutritional science in accessible terms
- Suggesting healthier alternatives
- Offering meal planning and portion guidance

Guidelines:
- Always consider the user's specific health conditions and restrictions
- Provide evidence-based nutritional advice
- Be encouraging and supportive while being honest about health impacts
- Suggest practical, actionable recommendations
- Explain complex nutritional concepts simply
- Reference recent food analysis results when relevant
- Recommend consulting healthcare providers for serious health concerns

Remember: You are not a replacement for medical advice. Always encourage users to consult their healthcare providers for medical decisions."""
    
    async def chat_with_context(
        self,
        user_message: str,
        user_context: UserHealthContext,
        chat_history: List[Dict[str, str]],
        analysis_context: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat with AI using full context including recent food analysis.
        
        Args:
            user_message: User's message
            user_context: User's health context
            chat_history: Previous chat messages
            analysis_context: Recent food analysis context
            provider: AI provider to use
            model: Specific model to use
            stream: Whether to stream response
            
        Returns:
            AI response with metadata
        """
        try:
            # Create contextualized system prompt
            system_prompt = self._create_contextualized_prompt(user_context, analysis_context)
            
            # Prepare messages
            messages = chat_history + [{"role": "user", "content": user_message}]
            
            # Get AI response
            if stream:
                return await self._stream_response(messages, system_prompt, provider, model)
            else:
                return await self._get_response(messages, system_prompt, provider, model)
                
        except Exception as e:
            logger.error(f"Chat service error: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        provider: str,
        model: Optional[str]
    ) -> Dict[str, Any]:
        """Get non-streaming AI response."""
        if provider == "openai":
            response = await self.ai_manager.chat_with_gpt4(
                messages=messages,
                system_prompt=system_prompt,
                model=model or "gpt-4-turbo-preview"
            )
        elif provider == "anthropic":
            # Convert messages to single prompt for Claude
            prompt = self._messages_to_prompt(messages, system_prompt)
            response = await self.ai_manager.analyze_with_claude(prompt=prompt)
        elif provider == "openrouter":
            response = await self.ai_manager.chat_with_openrouter(
                messages=messages,
                system_prompt=system_prompt,
                model=model or "meta-llama/llama-3.1-8b-instruct:free"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        response["timestamp"] = datetime.utcnow().isoformat()
        return response
    
    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        provider: str,
        model: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream AI response."""
        async for chunk in self.ai_manager.stream_chat_response(
            messages=messages,
            provider=provider,
            model=model,
            system_prompt=system_prompt
        ):
            yield {
                "chunk": chunk,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _create_contextualized_prompt(
        self,
        user_context: UserHealthContext,
        analysis_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create system prompt with user and analysis context."""
        prompt_parts = [self.default_system_prompt]
        
        # Add user health context
        if user_context:
            health_info = []
            
            # Primary health conditions
            if user_context.primary_profiles:
                conditions = [profile.profile_type.value for profile in user_context.primary_profiles]
                health_info.append(f"Health conditions: {', '.join(conditions)}")
            
            # Allergies
            if user_context.allergies:
                health_info.append(f"Allergies: {', '.join(user_context.allergies)}")
            
            # Dietary preferences
            if user_context.dietary_preferences:
                health_info.append(f"Dietary preferences: {', '.join(user_context.dietary_preferences)}")
            
            # Activity level and age group
            health_info.append(f"Activity level: {user_context.activity_level.value}")
            health_info.append(f"Age group: {user_context.age_group.value}")
            
            if health_info:
                prompt_parts.append(f"\nUSER HEALTH PROFILE:\n" + "\n".join(f"- {info}" for info in health_info))
        
        # Add recent analysis context
        if analysis_context:
            context_info = []
            
            if analysis_context.get("product_name"):
                context_info.append(f"Recently analyzed: {analysis_context['product_name']}")
            
            if analysis_context.get("overall_score") is not None:
                context_info.append(f"Health score: {analysis_context['overall_score']}/100")
            
            if analysis_context.get("safety_level"):
                context_info.append(f"Safety level: {analysis_context['safety_level']}")
            
            if analysis_context.get("key_recommendations"):
                recommendations = analysis_context["key_recommendations"][:3]  # Top 3
                context_info.append(f"Key recommendations: {'; '.join(recommendations)}")
            
            if context_info:
                prompt_parts.append(f"\nRECENT FOOD ANALYSIS:\n" + "\n".join(f"- {info}" for info in context_info))
        
        return "\n".join(prompt_parts)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Convert chat messages to single prompt for Claude."""
        prompt_parts = [system_prompt, "\n\nConversation:"]
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt_parts.append(f"\nHuman: {content}")
            elif role == "assistant":
                prompt_parts.append(f"\nAssistant: {content}")
        
        prompt_parts.append("\n\nAssistant:")
        return "\n".join(prompt_parts)
    
    async def generate_analysis_summary(
        self,
        analysis_result: Dict[str, Any],
        user_context: UserHealthContext
    ) -> str:
        """
        Generate a conversational summary of food analysis results.
        
        Args:
            analysis_result: Complete analysis result
            user_context: User's health context
            
        Returns:
            Conversational analysis summary
        """
        try:
            prompt = f"""
Based on this food analysis result, create a friendly, conversational summary for the user:

ANALYSIS RESULT:
- Product: {analysis_result.get('product_name', 'Unknown')}
- Overall Score: {analysis_result.get('overall_score', 0)}/100
- Safety Level: {analysis_result.get('safety_level', 'unknown')}
- Key Recommendations: {analysis_result.get('recommendations', [])}

USER HEALTH CONDITIONS:
{[profile.profile_type.value for profile in user_context.primary_profiles]}

Create a 2-3 sentence summary that:
1. Explains the overall assessment in simple terms
2. Highlights the most important point for their health conditions
3. Provides one actionable recommendation

Keep it conversational, supportive, and focused on what matters most to this user.
"""
            
            response = await self.ai_manager.chat_with_gpt4(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=200
            )
            
            return response["response"]
            
        except Exception as e:
            logger.error(f"Failed to generate analysis summary: {e}")
            return f"Analysis complete for {analysis_result.get('product_name', 'this product')}. Overall health score: {analysis_result.get('overall_score', 0)}/100."
    
    async def suggest_follow_up_questions(
        self,
        analysis_context: Dict[str, Any],
        user_context: UserHealthContext
    ) -> List[str]:
        """
        Suggest relevant follow-up questions based on analysis.
        
        Args:
            analysis_context: Recent analysis context
            user_context: User's health context
            
        Returns:
            List of suggested questions
        """
        try:
            prompt = f"""
Based on this food analysis, suggest 3-4 relevant follow-up questions a user might ask:

ANALYSIS CONTEXT:
- Product: {analysis_context.get('product_name', 'Unknown')}
- Score: {analysis_context.get('overall_score', 0)}/100
- Safety: {analysis_context.get('safety_level', 'unknown')}
- Health Conditions: {[profile.profile_type.value for profile in user_context.primary_profiles]}

Generate practical questions like:
- Portion size guidance
- Alternative product suggestions
- Preparation tips
- Health impact explanations
- Meal planning integration

Return only the questions, one per line, without numbering.
"""
            
            response = await self.ai_manager.chat_with_gpt4(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-turbo-preview",
                temperature=0.8,
                max_tokens=300
            )
            
            questions = [q.strip() for q in response["response"].split('\n') if q.strip()]
            return questions[:4]  # Limit to 4 questions
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return [
                "What's a good portion size for this product?",
                "Can you suggest healthier alternatives?",
                "How does this fit into my meal plan?",
                "What should I be most careful about with this product?"
            ]


# Global chat service instance
chat_service = ChatService()
