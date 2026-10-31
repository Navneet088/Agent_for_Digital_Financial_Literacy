"""Thin wrapper around IBM watsonx.ai (IBM Granite) chat inference."""
from __future__ import annotations

import logging
import time

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the IBM watsonx.ai call fails after retries."""


class GraniteClient:
    def __init__(self, settings: Settings, retries: int = 2):
        if not settings.has_ibm_credentials:
            raise LLMError("IBM_API_KEY and IBM_PROJECT_ID are required.")
        self._settings = settings
        self._retries = retries
        try:
            self._model = ModelInference(
                model_id=settings.model_id,
                credentials=Credentials(url=settings.ibm_url, api_key=settings.ibm_api_key),
                project_id=settings.ibm_project_id,
            )
        except Exception as exc:
            raise LLMError(f"Could not initialise IBM watsonx.ai client: {exc}") from exc

    def chat(self, messages: list[dict[str, str]]) -> str:
        params = {
            "max_tokens": self._settings.max_new_tokens,
            "temperature": self._settings.temperature,
        }
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = self._model.chat(messages=messages, params=params)
                content = resp["choices"][0]["message"]["content"]
                return (content or "").strip()
            except Exception as exc:  # network / quota / auth errors
                last_exc = exc
                log.warning("Granite call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self._retries:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"IBM watsonx.ai request failed: {last_exc}") from last_exc
