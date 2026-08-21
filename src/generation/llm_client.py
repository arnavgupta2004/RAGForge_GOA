"""Claude API wrapper for grounded generation.

One retry on transient failures (timeout, rate limit, 5xx), then a typed
GenerationError the pipeline turns into a safe failure response -- generation
never raises a raw SDK exception up to the API layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.models import ContextItem


class GenerationError(Exception):
    pass


@dataclass
class GenerationResult:
    answer: str
    model: str
    stop_reason: str | None


_RETRYABLE = (
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


class LLMClient:
    def __init__(self, model: str, max_tokens: int, temperature: float, timeout_s: float, max_retries: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise GenerationError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s)

    async def generate(self, query: str, context: list[ContextItem]) -> GenerationResult:
        user_prompt = build_user_prompt(query, context)

        @retry(
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )
        async def _call() -> anthropic.types.Message:
            return await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

        try:
            message = await _call()
        except _RETRYABLE as e:
            raise GenerationError(f"LLM request failed after retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise GenerationError(f"LLM request rejected: {e}") from e

        text_parts = [block.text for block in message.content if block.type == "text"]
        answer = "".join(text_parts).strip()
        if not answer:
            raise GenerationError("LLM returned an empty response")

        return GenerationResult(answer=answer, model=self.model, stop_reason=message.stop_reason)
