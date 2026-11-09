"""
Official Google Gen AI SDK (`google-genai`) wrapper for the
Tulane University Faculty Onboarding Assistant.

Responsibilities:
- Reads `GEMINI_API_KEY` and `GEMINI_MODEL` from the project-root `.env` file.
- Never invents a model ID and never prints or logs API keys.
- Configures explicit HTTP request timeouts via `types.HttpOptions(timeout=...)` (milliseconds).
- Uses structured JSON output (`response_mime_type="application/json"`, `response_schema=...`).
- Applies bounded retries with exponential backoff only for transient/rate-limit errors;
  never retries authentication, permission, or invalid-model errors endlessly.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

PLACEHOLDER_VALUES = {
    "",
    "your_gemini_api_key_here",
    "your_available_gemini_model_id_here",
    "changeme",
    "replace_me",
}

T = TypeVar("T", bound=BaseModel)


class LLMConfigurationError(RuntimeError):
    """Raised when GEMINI_API_KEY or GEMINI_MODEL is missing or set to a placeholder."""


class LLMServiceError(RuntimeError):
    """Raised when the hosted Gemini API fails (auth, quota, timeout, or network error)."""

    def __init__(self, message: str, is_transient: bool = False, status_code: int | None = None) -> None:
        super().__init__(message)
        self.is_transient = is_transient
        self.status_code = status_code


def _redact_secrets(text: str, api_key: str | None = None) -> str:
    """Ensure API keys are never exposed in error messages or logs."""
    cleaned = text
    if api_key and len(api_key) >= 6:
        cleaned = cleaned.replace(api_key, "[REDACTED_API_KEY]")
    # Also redact any AIza... style Google API key pattern
    cleaned = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", cleaned)
    return cleaned


@dataclass(frozen=True)
class GeminiSettings:
    """Validated configuration loaded from `.env` / environment."""

    api_key: str
    model_id: str
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __repr__(self) -> str:
        return (
            f"GeminiSettings(api_key='[REDACTED]', model_id={self.model_id!r}, "
            f"timeout_seconds={self.timeout_seconds}, max_retries={self.max_retries})"
        )


def load_gemini_settings(env_path: Path = DEFAULT_ENV_PATH) -> GeminiSettings:
    """
    Load and validate `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env` or environment variables.
    Raises `LLMConfigurationError` with clear setup instructions if missing or placeholder.
    """
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

    raw_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    raw_model = (os.environ.get("GEMINI_MODEL") or "").strip()

    missing_items: list[str] = []
    if raw_key.lower() in PLACEHOLDER_VALUES:
        missing_items.append("GEMINI_API_KEY")
    if raw_model.lower() in PLACEHOLDER_VALUES:
        missing_items.append("GEMINI_MODEL")

    if missing_items:
        missing_str = ", ".join(missing_items)
        raise LLMConfigurationError(
            f"Missing or placeholder Gemini configuration: {missing_str}.\n"
            "Setup Instructions:\n"
            "  1. Copy `.env.example` to `.env` in the project root:\n"
            "       Copy-Item .env.example .env\n"
            "  2. Obtain a Gemini API key from Google AI Studio (https://aistudio.google.com/apikey)\n"
            "     and set `GEMINI_API_KEY=<your_key>` in `.env`.\n"
            "  3. Set `GEMINI_MODEL=<your_model_id>` in `.env` to a text-generation model\n"
            "     available to your account (see `.env.example` for the command to list models).\n"
            "  Note: You can still run retrieval-only search without an API key using:\n"
            "       .\\.venv\\Scripts\\python backend\\search_demo.py"
        )

    try:
        timeout_sec = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout_sec = 30.0

    try:
        max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "2"))
    except ValueError:
        max_retries = 2

    return GeminiSettings(
        api_key=raw_key,
        model_id=raw_model,
        timeout_seconds=max(1.0, timeout_sec),
        max_retries=max(0, max_retries),
    )


def _classify_exception(exc: Exception, api_key: str | None = None) -> tuple[bool, int | None, str]:
    """
    Determine whether an exception from `google-genai` / `httpx` is transient (retryable)
    or permanent (auth/permission/invalid model), and return a safe error description.
    """
    safe_msg = _redact_secrets(str(exc), api_key=api_key)
    status_code: int | None = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        status_code = code if isinstance(code, int) else status_code
        if status_code in (401, 403):
            return (
                False,
                status_code,
                f"Gemini API authentication/permission failed (HTTP {status_code}). "
                "Please verify `GEMINI_API_KEY` in `.env`.",
            )
        if status_code == 404:
            return (
                False,
                status_code,
                f"Configured `GEMINI_MODEL` was not found or is not supported (HTTP 404): {safe_msg}",
            )
        if status_code == 400:
            if "API_KEY_INVALID" in safe_msg or "api key not valid" in safe_msg.lower():
                return (
                    False,
                    400,
                    "Gemini API key is invalid (HTTP 400). Please check `GEMINI_API_KEY` in `.env`.",
                )
            return (
                False,
                400,
                f"Gemini API rejected the request (HTTP 400): {safe_msg}",
            )
        if status_code in (408, 429, 500, 502, 503, 504):
            return (
                True,
                status_code,
                f"Gemini API transient/rate-limit error (HTTP {status_code}): {safe_msg}",
            )

    lower_msg = safe_msg.lower()
    if any(k in lower_msg for k in ("401", "403", "unauthorized", "permission_denied", "api_key_invalid", "invalid api key")):
        return (
            False,
            status_code or 401,
            f"Gemini API authentication error: {safe_msg}",
        )
    if any(k in lower_msg for k in ("timeout", "timed out", "rate limit", "429", "503", "temporarily unavailable", "connection")):
        return (
            True,
            status_code,
            f"Gemini API request timed out or encountered a transient error: {safe_msg}",
        )

    return False, status_code, f"Gemini API error: {safe_msg}"


class GeminiStructuredClient:
    """
    Wrapper around `google.genai.Client` providing structured Pydantic/JSON generation
    with explicit timeouts and bounded transient retries.
    """

    def __init__(
        self,
        settings: GeminiSettings | None = None,
        env_path: Path = DEFAULT_ENV_PATH,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings if settings is not None else load_gemini_settings(env_path=env_path)
        self._sleep_fn = sleep_fn
        # `HttpOptions.timeout` in `google-genai` is specified in milliseconds
        timeout_ms = int(self.settings.timeout_seconds * 1000)
        self._client = genai.Client(
            api_key=self.settings.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    @property
    def model_id(self) -> str:
        return self.settings.model_id

    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        """
        Call `client.models.generate_content` with `response_mime_type="application/json"`
        and `response_schema`, returning a validated Pydantic model instance `T`.
        Retries up to `self.settings.max_retries` times ONLY on transient/rate-limit errors.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
        )

        max_attempts = 1 + self.settings.max_retries
        last_error: LLMServiceError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.settings.model_id,
                    contents=user_prompt,
                    config=config,
                )
                raw_text = getattr(response, "text", None)
                if not raw_text or not str(raw_text).strip():
                    raise ValueError("Model returned an empty response body.")

                parsed_dict = json.loads(str(raw_text))
                return response_schema.model_validate(parsed_dict)

            except (json.JSONDecodeError, ValueError) as parse_exc:
                # Raise ValueError so caller's structure-retry handler can handle malformed JSON
                raise ValueError(
                    f"Model output did not match expected structured schema: {parse_exc}"
                ) from parse_exc
            except Exception as api_exc:
                is_transient, code, safe_desc = _classify_exception(
                    api_exc, api_key=self.settings.api_key
                )
                last_error = LLMServiceError(
                    safe_desc, is_transient=is_transient, status_code=code
                )
                if not is_transient or attempt >= max_attempts:
                    raise last_error from api_exc

                backoff_sec = 1.0 * (2 ** (attempt - 1))
                self._sleep_fn(backoff_sec)

        assert last_error is not None
        raise last_error
