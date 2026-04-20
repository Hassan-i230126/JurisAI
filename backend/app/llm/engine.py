"""
Juris AI — LLM Engine
Ollama-based LLM wrapper with streaming generation via phi4-mini.
"""

import json
import time
from typing import AsyncGenerator, Optional, List, Dict

import httpx
from loguru import logger

from app.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_NUM_CTX,
    LLM_NUM_THREADS,
)


class LLMEngine:
    """
    Wrapper around Ollama for streaming LLM generation.
    
    Uses phi4-mini with carefully tuned parameters for deterministic
    legal responses. Includes a post-generation hallucination guard.
    """

    def __init__(self):
        """Initialize the LLM engine."""
        self._model = LLM_MODEL
        self._options = {
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
            "num_predict": LLM_MAX_TOKENS,
            "num_ctx": LLM_NUM_CTX,
            "num_thread": LLM_NUM_THREADS,
            "repeat_penalty": 1.1,
        }
        self._is_available = False
        self._http_timeout = httpx.Timeout(connect=2.0, read=None, write=30.0, pool=10.0)

    async def check_availability(self) -> bool:
        """
        Check that Ollama is running and the model is available.
        
        Returns:
            True if Ollama is reachable and the model is pulled.
        """
        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=httpx.Timeout(connect=2.0, read=8.0, write=8.0, pool=8.0)) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                models_response = response.json()

            # Check if our model is in the list.
            available_models = [
                m.get("name", m.get("model", ""))
                for m in models_response.get("models", [])
            ]

            # Check for model name (may include tag like ":latest")
            model_found = any(
                self._model in m or m.startswith(self._model)
                for m in available_models
            )

            if not model_found:
                logger.critical(
                    "LLM model '{}' not found. Available models: {}. "
                    "Run: ollama pull {} && ollama pull bge-m3",
                    self._model, available_models, self._model
                )
                self._is_available = False
                return False

            self._is_available = True
            logger.info("LLM engine ready | model={} | options={}", self._model, self._options)
            return True

        except Exception as e:
            logger.critical(
                "Cannot connect to Ollama at {}. Error: {}. "
                "Run: ollama serve",
                OLLAMA_BASE_URL, str(e)
            )
            self._is_available = False
            return False

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the LLM one by one.
        
        Uses ollama.chat() with stream=True, wrapped in an async
        generator. Each yielded value is a single token string.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Yields:
            Individual token strings as they are generated.
        """
        start_time = time.time()
        total_tokens = 0

        try:
            payload = {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": self._options,
            }

            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=self._http_timeout) as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            total_tokens += 1
                            yield token

                        if chunk.get("done"):
                            break

        except Exception as e:
            self._is_available = False
            logger.error("LLM generation error: {}", str(e))
            yield f"\n\n⚠️ LLM generation error: {str(e)}"

        finally:
            elapsed = time.time() - start_time
            logger.info(
                "LLM generation complete | tokens={} | time_ms={:.0f}",
                total_tokens, elapsed * 1000
            )

    async def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate a complete response (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Returns:
            The complete generated text.
        """
        tokens = []
        async for token in self.generate_stream(messages):
            tokens.append(token)
        return "".join(tokens)

    def apply_hallucination_guard(
        self,
        response: str,
        rag_found_nothing: bool,
    ) -> str:
        """
        Post-generation hallucination guard.
        
        If RAG found nothing AND the response doesn't contain the
        uncertainty phrase, append a disclaimer.
        
        Args:
            response: The full generated response text.
            rag_found_nothing: Whether the retriever returned empty results.
            
        Returns:
            The response, possibly with an appended disclaimer.
        """
        uncertainty_phrase = "I do not have sufficient information"

        if rag_found_nothing and uncertainty_phrase not in response:
            logger.warning("Hallucination guard triggered — appending disclaimer")
            response += (
                "\n\n⚠️ Note: This response was generated without retrieved "
                "legal sources. Please verify independently."
            )

        return response

    @property
    def is_available(self) -> bool:
        """Whether the LLM engine is available for generation."""
        return self._is_available
