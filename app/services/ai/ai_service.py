"""
AI service integrations for OpenAI, Claude, and OpenRouter.
Provides chat functionality and advanced nutritional analysis.
"""

import openai
import anthropic
import httpx
from typing import Dict, Any, List, Optional, AsyncGenerator
import structlog
import json

from app.core.config import settings
from app.cache.cache_manager import cache_manager

logger = structlog.get_logger(__name__)


class AIServiceManager:
    """Centralized AI service manager for multiple providers."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AI service clients."""
        try:
            if settings.openai_api_key:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            
            if settings.anthropic_api_key:
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            
            if settings.openrouter_api_key:
                logger.info("OpenRouter API key configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI clients: {e}")
    
    async def chat_with_gpt4(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Chat with OpenAI GPT-4 for conversational AI.
        
        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt
            model: GPT model to use
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
            
        Returns:
            AI response with metadata
        """
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            # Prepare messages
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            # Make API call
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            ai_message = response.choices[0].message.content
            
            return {
                "response": ai_message,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"Failed to get response from GPT-4: {str(e)}")
    
    async def analyze_with_claude(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Use Claude for advanced nutritional analysis and reasoning.
        
        Args:
            prompt: Analysis prompt
            model: Claude model to use
            max_tokens: Maximum response length
            temperature: Response creativity (0-1)
            
        Returns:
            Analysis response with metadata
        """
        if not self.anthropic_client:
            raise Exception("Anthropic client not initialized")
        
        try:
            # Make API call
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response
            ai_response = response.content[0].text
            
            return {
                "response": ai_response,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Failed to get response from Claude: {str(e)}")
    
    async def chat_with_openrouter(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Chat with models via OpenRouter (including free Llama models).
        
        Args:
            messages: List of chat messages
            model: Model to use (e.g., "meta-llama/llama-3.1-8b-instruct:free")
            system_prompt: Optional system prompt
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
            
        Returns:
            AI response with metadata
        """
        if not settings.openrouter_api_key:
            raise Exception("OpenRouter API key not configured")
        
        try:
            # Prepare messages
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://wellix.com",
                "X-Title": "Wellix AI Food Analysis"
            }
            
            payload = {
                "model": model,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract response
                ai_message = data["choices"][0]["message"]["content"]
                
                return {
                    "response": ai_message,
                    "model": model,
                    "usage": data.get("usage", {}),
                    "finish_reason": data["choices"][0].get("finish_reason")
                }
                
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise Exception(f"Failed to get response from OpenRouter: {str(e)}")
    
    async def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        provider: str = "openai",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response for real-time UI updates.
        
        Args:
            messages: List of chat messages
            provider: AI provider ("openai", "anthropic", "openrouter")
            model: Specific model to use
            system_prompt: Optional system prompt
            
        Yields:
            Streaming response chunks
        """
        if provider == "openai" and self.openai_client:
            async for chunk in self._stream_openai_response(messages, model, system_prompt):
                yield chunk
        elif provider == "openrouter":
            async for chunk in self._stream_openrouter_response(messages, model, system_prompt):
                yield chunk
        else:
            # Fallback to non-streaming
            if provider == "anthropic":
                response = await self.analyze_with_claude(messages[-1]["content"])
            elif provider == "openrouter":
                response = await self.chat_with_openrouter(messages, model, system_prompt)
            else:
                response = await self.chat_with_gpt4(messages, system_prompt, model or "gpt-4-turbo-preview")
            
            yield response["response"]
    
    async def _stream_openai_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI response."""
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            stream = await self.openai_client.chat.completions.create(
                model=model or "gpt-4-turbo-preview",
                messages=formatted_messages,
                stream=True,
                temperature=0.7
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"
    
    async def _stream_openrouter_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream OpenRouter response."""
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            headers = {
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://wellix.com",
                "X-Title": "Wellix AI Food Analysis"
            }
            
            payload = {
                "model": model or "meta-llama/llama-3.1-8b-instruct:free",
                "messages": formatted_messages,
                "stream": True,
                "temperature": 0.7
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            yield f"Error: {str(e)}"


class NutritionAnalysisAI:
    """Specialized AI service for nutrition analysis tasks."""
    
    def __init__(self, ai_manager: AIServiceManager):
        self.ai_manager = ai_manager
    
    async def enhance_nutrition_analysis(
        self,
        nutrition_data: Dict[str, Any],
        health_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Claude to enhance nutrition analysis with advanced reasoning.
        
        Args:
            nutrition_data: Extracted nutrition information
            health_context: User's health context
            
        Returns:
            Enhanced analysis with AI insights
        """
        prompt = self._create_analysis_prompt(nutrition_data, health_context)
        
        try:
            response = await self.ai_manager.analyze_with_claude(
                prompt=prompt,
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse structured response
            analysis = self._parse_analysis_response(response["response"])
            
            return {
                "ai_insights": analysis,
                "model_used": response["model"],
                "usage": response["usage"]
            }
            
        except Exception as e:
            logger.error(f"AI nutrition analysis failed: {e}")
            return {"ai_insights": {}, "error": str(e)}
    
    def _create_analysis_prompt(
        self,
        nutrition_data: Dict[str, Any],
        health_context: Dict[str, Any]
    ) -> str:
        """Create structured prompt for nutrition analysis."""
        return f"""
As a nutrition expert AI, analyze this food product for the given health context.

PRODUCT NUTRITION DATA:
{json.dumps(nutrition_data, indent=2)}

USER HEALTH CONTEXT:
{json.dumps(health_context, indent=2)}

Please provide a comprehensive analysis including:

1. NUTRITIONAL ASSESSMENT:
   - Overall nutritional quality
   - Key strengths and concerns
   - Macronutrient balance evaluation

2. HEALTH IMPACT:
   - Specific impacts for user's health conditions
   - Potential benefits and risks
   - Interaction with medications (if applicable)

3. RECOMMENDATIONS:
   - Portion size guidance
   - Timing recommendations
   - Preparation suggestions
   - Alternative product suggestions

4. SCIENTIFIC REASONING:
   - Evidence-based explanations
   - Relevant nutritional science
   - Health condition considerations

Format your response as structured JSON with clear sections.
"""
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse AI analysis response into structured data."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to text parsing
                return {
                    "analysis_text": response,
                    "structured": False
                }
                
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return {
                "analysis_text": response,
                "parse_error": str(e)
            }


# Global AI service instances
ai_service_manager = AIServiceManager()
nutrition_analysis_ai = NutritionAnalysisAI(ai_service_manager)
