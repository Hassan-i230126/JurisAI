"""
Juris AI — Shared GeminiJudge for DeepEval
==========================================
DeepEval 3.x requires a custom LLM judge to implement `generate()` with an
optional `schema` parameter (a Pydantic BaseModel subclass).  When `schema`
is supplied, DeepEval wants a *structured* response — not a plain string.
Without this, every metric crashes with:
    AttributeError: 'str' object has no attribute 'get'

Additionally, the Gemini Flash free tier allows ~15 RPM.  We add exponential
back-off so that 429 RESOURCE_EXHAUSTED errors are retried automatically.

Usage in test files:
    from gemini_judge import GeminiJudge
    judge = GeminiJudge()
"""

from __future__ import annotations

import json
import os
import time
import re
import threading
from typing import Optional, Type, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from google import genai as google_genai
from google.genai import types as genai_types
from pydantic import BaseModel


# ── Rate-limit settings ───────────────────────────────────────────────────────
_MAX_RETRIES   = 5      # maximum retry attempts on 429
_BASE_DELAY    = 6.0    # seconds between EVERY API call (keeps us under 30 RPM for flash-lite)
_RETRY_DELAY   = 70.0   # seconds to wait after a 429 (Gemini hints 57s; we use 70s to be safe)
_API_LOCK      = threading.Lock()

def _call_with_retry(fn, *args, **kwargs):
    """Call `fn(*args, **kwargs)`, retrying on 429 with back-off. Uses a global lock to prevent concurrent calls."""
    for attempt in range(_MAX_RETRIES):
        try:
            with _API_LOCK:
                time.sleep(_BASE_DELAY)  # throttle every call to stay under rate limit
                return fn(*args, **kwargs)
        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = _RETRY_DELAY * (2 ** attempt)
                print(f"\n[GeminiJudge] Rate limit hit — waiting {wait:.0f}s before retry {attempt+1}/{_MAX_RETRIES}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"[GeminiJudge] All {_MAX_RETRIES} retries exhausted.")


def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from a string that may contain markdown
    fences or surrounding prose (common with LLM responses).
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    brace = re.search(r"\{[\s\S]+\}", text)
    if brace:
        try:
            return json.loads(brace.group())
        except json.JSONDecodeError:
            pass

    return {}


class GeminiJudge(DeepEvalBaseLLM):
    """
    Google Gemini Flash (google-genai SDK) wrapped as a DeepEval judge.

    Key differences from a naive wrapper:
    - `generate(prompt, schema)` — when `schema` is a Pydantic BaseModel subclass,
      this method returns an *instance* of that schema.  DeepEval 3.x requires
      this for all metrics that use structured output internally.
    - Rate-limit retry with exponential back-off to handle 429 from free tier.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/app/apikey "
                "and add it to your .env file as  GEMINI_API_KEY=AIza..."
            )
        self._client = google_genai.Client(api_key=api_key)

    # ── DeepEvalBaseLLM required interface ────────────────────────────────────

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        return self._client

    # ── Core generation — handles both plain-text and structured-output calls ─

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        DeepEval 3.x calls this with `schema` set to a Pydantic model when it
        needs structured output (which is what all built-in metrics require).
        We ask Gemini to respond as JSON and then parse it into the schema.
        """
        if schema is not None:
            return self._generate_structured(prompt, schema)
        return self._generate_text(prompt)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """Async version — delegates to the sync implementation (no async SDK call needed)."""
        return self.generate(prompt, schema)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_text(self, prompt: str) -> str:
        """Plain text generation."""
        def _call():
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return resp.text

        return _call_with_retry(_call)

    def _generate_structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        """
        Structured generation: ask Gemini for a JSON response and parse it
        into the requested Pydantic schema.

        Strategy:
        1. Try with response_mime_type="application/json" (native structured output)
        2. If that fails (e.g. unsupported model version), ask in the prompt and parse
        """
        # Strategy 1: native JSON mode
        def _call_json():
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return resp.text

        try:
            raw = _call_with_retry(_call_json)
            data = _extract_json(raw)
            return schema(**data)
        except Exception:
            pass

        # Strategy 2: prompt-based JSON extraction
        json_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema: "
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Do not include any explanation or markdown. Output only the JSON."
        )

        def _call_text():
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=json_prompt,
            )
            return resp.text

        raw = _call_with_retry(_call_text)
        data = _extract_json(raw)

        # Fill missing required fields with safe defaults
        try:
            return schema(**data)
        except Exception:
            # Build a minimal instance with defaults for all fields
            fields = schema.model_fields
            safe = {}
            for field_name, field_info in fields.items():
                if field_name in data:
                    safe[field_name] = data[field_name]
                else:
                    annotation = field_info.annotation
                    if annotation is str or annotation == Optional[str]:
                        safe[field_name] = ""
                    elif annotation is float or annotation == Optional[float]:
                        safe[field_name] = 0.0
                    elif annotation is int or annotation == Optional[int]:
                        safe[field_name] = 0
                    elif annotation is bool or annotation == Optional[bool]:
                        safe[field_name] = False
                    elif annotation is list or (hasattr(annotation, "__origin__") and annotation.__origin__ is list):
                        safe[field_name] = []
                    else:
                        safe[field_name] = None
            return schema(**safe)
