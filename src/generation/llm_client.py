"""Gemini API wrapper for grounded generation.

Uses google-genai's async client (client.aio.models.generate_content).
One retry on transient failures (5xx server errors, 429 rate limit,
network timeout/connect errors), then a typed GenerationError the pipeline
turns into a safe failure response -- generation never raises a raw SDK
exception up to the API layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.models import ContextItem


class GenerationError(Exception):
    pass


@dataclass
class GenerationResult:
    answer: str
    model: str
    stop_reason: str | None


def _is_retryable(e: BaseException) -> bool:
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(e, errors.ServerError):
        return True
    if isinstance(e, errors.ClientError) and getattr(e, "code", None) == 429:
        return True
    return False


class LLMClient:
    def __init__(self, model: str, max_tokens: int, temperature: float, timeout_s: float, max_retries: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GenerationError("GEMINI_API_KEY is not set")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    async def generate(self, query: str, context: list[ContextItem]) -> GenerationResult:
        user_prompt = build_user_prompt(query, context)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        @retry(
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_fixed(0.5),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        async def _call() -> types.GenerateContentResponse:
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )

        try:
            response = await _call()
        except (errors.ServerError, errors.ClientError, httpx.TimeoutException, httpx.ConnectError) as e:
            raise GenerationError(f"LLM request failed: {e}") from e

        answer = (response.text or "").strip()
        if not answer:
            raise GenerationError("LLM returned an empty response")

        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates[0].finish_reason else None

        return GenerationResult(answer=answer, model=self.model, stop_reason=finish_reason)
