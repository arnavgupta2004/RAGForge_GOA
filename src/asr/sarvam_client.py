"""Sarvam AI speech-to-text client.

POST https://api.sarvam.ai/speech-to-text, multipart/form-data, auth via the
`api-subscription-key` header (see docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe).

Uses model=saaras:v3 with mode="translate": for non-English speech this
transcribes AND translates to English in a single call, which is what feeds
retrieval (indexed in English -- see docs/decisions.md on language choice).
For English speech, translate mode passes the transcript through unchanged.
The detected `language_code` is still returned so the UI can show what
language was actually spoken.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

_ENDPOINT = "https://api.sarvam.ai/speech-to-text"

_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError)


class ASRError(Exception):
    pass


@dataclass
class TranscriptionResult:
    transcript: str
    language_code: str | None
    language_probability: float | None
    request_id: str | None


class SarvamClient:
    def __init__(self, model: str, mode: str, timeout_s: float, max_retries: int) -> None:
        self.model = model
        self.mode = mode
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.api_key = os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise ASRError("SARVAM_API_KEY is not set")

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> TranscriptionResult:
        headers = {"api-subscription-key": self.api_key}
        data = {"model": self.model, "mode": self.mode, "language_code": "unknown"}
        files = {"file": (filename, audio_bytes)}

        @retry(
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )
        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                return await client.post(_ENDPOINT, headers=headers, data=data, files=files)

        try:
            response = await _call()
        except _RETRYABLE as e:
            raise ASRError(f"Sarvam request failed after retries: {e}") from e

        if response.status_code != 200:
            raise ASRError(f"Sarvam returned {response.status_code}: {response.text[:300]}")

        payload = response.json()
        transcript = (payload.get("transcript") or "").strip()
        if not transcript:
            raise ASRError("Sarvam returned an empty transcript")

        return TranscriptionResult(
            transcript=transcript,
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            request_id=payload.get("request_id"),
        )
