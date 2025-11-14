"""
LLM Client interface.

Factor 1: Natural language to tool calls.
This interface handles communication with LLM providers and converts
natural language to structured tool calls.
"""

from typing import Dict, List, Any, Optional
import aiohttp
import json
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Abstract interface for LLM providers.

    Supports multiple backends: OpenAI, Anthropic, RunPod, Ollama, OpenRouter
    """

    def __init__(
        self,
        provider: str = "openai",
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        self.provider = provider
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.site_url = site_url  # For OpenRouter attribution
        self.site_name = site_name  # For OpenRouter attribution
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Factor 1: Converts natural language to tool calls.

        Returns:
            {
                "content": str,  # Text response
                "tool_calls": List[Dict],  # Structured tool calls
                "finish_reason": str
            }
        """
        if self.provider == "openai":
            return self._generate_openai(
                prompt, system_prompt, tools, temperature, max_tokens
            )
        elif self.provider == "anthropic":
            return self._generate_anthropic(
                prompt, system_prompt, tools, temperature, max_tokens
            )
        elif self.provider == "runpod":
            return self._generate_runpod(
                prompt, system_prompt, tools, temperature, max_tokens
            )
        elif self.provider == "ollama":
            return self._generate_ollama(
                prompt, system_prompt, tools, temperature, max_tokens
            )
        elif self.provider == "openrouter":
            return self._generate_openrouter(
                prompt, system_prompt, tools, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        # This would use the actual OpenAI SDK
        # For now, returning a mock response
        logger.info(f"Generating with OpenAI: {prompt[:50]}...")

        # Mock response - in real implementation, call OpenAI API
        return {
            "content": "I will help you with that task.",
            "tool_calls": [],
            "finish_reason": "stop"
        }

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using Anthropic API."""
        logger.info(f"Generating with Anthropic: {prompt[:50]}...")

        # Mock response - in real implementation, call Anthropic API
        return {
            "content": "I will help you with that task.",
            "tool_calls": [],
            "finish_reason": "end_turn"
        }

    def _generate_runpod(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using RunPod endpoint."""
        logger.info(f"Generating with RunPod: {prompt[:50]}...")

        # Format prompt for RunPod
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Mock response - in real implementation, call RunPod API
        return {
            "content": "I will help you with that task.",
            "tool_calls": [],
            "finish_reason": "stop"
        }

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using local Ollama."""
        logger.info(f"Generating with Ollama: {prompt[:50]}...")

        # Format for Ollama
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Mock response - in real implementation, call Ollama API
        return {
            "content": "I will help you with that task.",
            "tool_calls": [],
            "finish_reason": "stop"
        }

    def _generate_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Generate using OpenRouter API.

        OpenRouter provides unified access to hundreds of AI models through
        an OpenAI-compatible API endpoint.

        Endpoint: https://openrouter.ai/api/v1/chat/completions

        Features:
        - Automatic fallbacks and cost-effective model selection
        - Access to 100+ models from multiple providers
        - OpenAI-compatible API format
        - Optional attribution headers (HTTP-Referer, X-Title)
        """
        logger.info(f"Generating with OpenRouter ({self.model}): {prompt[:50]}...")

        # OpenRouter endpoint
        endpoint = self.endpoint or "https://openrouter.ai/api/v1"

        # Mock response - in real implementation, call OpenRouter API
        # Real implementation would use OpenAI SDK with custom base_url:
        #
        # from openai import OpenAI
        # client = OpenAI(
        #     base_url=endpoint,
        #     api_key=self.api_key,
        # )
        # extra_headers = {}
        # if self.site_url:
        #     extra_headers["HTTP-Referer"] = self.site_url
        # if self.site_name:
        #     extra_headers["X-Title"] = self.site_name
        #
        # completion = client.chat.completions.create(
        #     extra_headers=extra_headers,
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": system_prompt} if system_prompt else None,
        #         {"role": "user", "content": prompt}
        #     ],
        #     tools=tools,
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )

        return {
            "content": "I will help you with that task.",
            "tool_calls": [],
            "finish_reason": "stop"
        }


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.

    Factor 1: Demonstrates tool calling behavior.
    """

    def __init__(self):
        super().__init__(provider="mock")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate mock response with tool calls."""

        # Simulate the agent making progress
        if "authentication" in prompt.lower() or "auth" in prompt.lower():
            return {
                "content": "I'll create an authentication API with JWT tokens.",
                "tool_calls": [
                    {
                        "name": "write_file",
                        "arguments": {
                            "path": "auth_api.py",
                            "content": "# Authentication API implementation\n# TODO: Add JWT logic"
                        }
                    }
                ],
                "finish_reason": "tool_calls"
            }
        elif "complete" in prompt.lower():
            return {
                "content": "Task completed successfully.",
                "tool_calls": [
                    {
                        "name": "complete_task",
                        "arguments": {
                            "summary": "Completed the requested task",
                            "artifacts": []
                        }
                    }
                ],
                "finish_reason": "tool_calls"
            }
        else:
            return {
                "content": "I'm working on this task. Let me make progress.",
                "tool_calls": [],
                "finish_reason": "stop"
            }
