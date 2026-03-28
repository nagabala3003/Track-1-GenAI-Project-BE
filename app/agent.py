import logging
import os
import re
import time
from typing import Final, Literal, Optional, TypedDict

from google import genai

GEMINI_API_KEY_ENVIRONMENT_VARIABLE: Final[str] = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL_NAME: Final[str] = "gemini-flash-latest"
GEMINI_MODEL_CANDIDATES_ENVIRONMENT_VARIABLE: Final[str] = "GEMINI_MODEL_CANDIDATES"
DEFAULT_MODEL_FALLBACK_ORDER: Final[tuple[str, ...]] = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
)
logger: logging.Logger = logging.getLogger(__name__)

api_key: str = os.getenv(GEMINI_API_KEY_ENVIRONMENT_VARIABLE, "")
if not api_key:
    raise ValueError(
        f"{GEMINI_API_KEY_ENVIRONMENT_VARIABLE} is missing. Set it before running the app."
    )

client = genai.Client(api_key=api_key)

MAX_LOGGED_TEXT_LENGTH: Final[int] = 500

GeminiApiStatus = Literal["success", "error"]


class GeminiApiResponse(TypedDict):
    status: GeminiApiStatus
    data: Optional[str]
    error: Optional[str]


def trim_text_for_log(text: str, max_length: int = MAX_LOGGED_TEXT_LENGTH) -> str:
    """Trim long text for logs to avoid flooding output."""
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}...[truncated]"


def normalize_model_name(model_name: str) -> str:
    clean = model_name.strip()
    if clean.startswith("models/"):
        return clean.split("/", 1)[1]
    return clean


def supports_generate_content(model: object) -> bool:
    methods = (
        getattr(model, "supported_generation_methods", None)
        or getattr(model, "supportedMethods", None)
        or []
    )
    lowered = {str(method).strip().lower() for method in methods}
    return "generatecontent" in lowered


def discover_supported_models() -> list[str]:
    discovered: list[str] = []
    try:
        for model in client.models.list():
            raw_name = str(getattr(model, "name", "")).strip()
            if not raw_name:
                continue
            if not supports_generate_content(model):
                continue
            model_name = normalize_model_name(raw_name)
            discovered.append(model_name)
    except Exception as error:
        logger.warning("Failed to discover Gemini models: %s", trim_text_for_log(str(error)))
        return []

    def model_priority(name: str) -> tuple[int, str]:
        lowered = name.lower()
        if "flash" in lowered:
            return (0, lowered)
        if "pro" in lowered:
            return (1, lowered)
        return (2, lowered)

    return sorted(discovered, key=model_priority)


def parse_model_candidates_from_env() -> list[str]:
    raw_models: str = os.getenv(GEMINI_MODEL_CANDIDATES_ENVIRONMENT_VARIABLE, "")
    if raw_models.strip():
        requested = [normalize_model_name(model) for model in raw_models.split(",") if model.strip()]
    else:
        requested = [DEFAULT_GEMINI_MODEL_NAME, *DEFAULT_MODEL_FALLBACK_ORDER]

    discovered = discover_supported_models()
    requested.extend(discovered)

    deduped: list[str] = []
    seen: set[str] = set()
    for model_name in requested:
        key = normalize_model_name(model_name).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalize_model_name(model_name))

    if deduped:
        logger.info("Gemini model candidates: %s", ", ".join(deduped[:8]))
    return deduped


def is_quota_error(error_message: str) -> bool:
    text = error_message.upper()
    return "429" in text and "RESOURCE_EXHAUSTED" in text


def parse_retry_delay_seconds(error_message: str) -> Optional[float]:
    match = re.search(r"retry in\s+([0-9]*\.?[0-9]+)s", error_message, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def generate_response(prompt: str) -> GeminiApiResponse:
    """Generate a response from Gemini for the given prompt."""
    if not prompt or not prompt.strip():
        return {"status": "error", "data": None, "error": "Input prompt must not be empty."}

    model_candidates: list[str] = parse_model_candidates_from_env()
    last_error_message: Optional[str] = None
    saw_quota_exhaustion: bool = False
    retry_delay_seconds: Optional[float] = None

    def try_models() -> Optional[GeminiApiResponse]:
        nonlocal last_error_message, saw_quota_exhaustion, retry_delay_seconds

        for model_name in model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                response_text: Optional[str] = getattr(response, "text", None)
                if response_text and response_text.strip():
                    return {"status": "success", "data": response_text, "error": None}

                logger.warning("Gemini returned empty response for model=%s", model_name)
                last_error_message = f"Gemini returned an empty response for model={model_name}."
            except Exception as error:
                error_message: str = str(error) or error.__class__.__name__
                last_error_message = error_message

                if is_quota_error(error_message):
                    saw_quota_exhaustion = True
                    retry_after = parse_retry_delay_seconds(error_message)
                    if retry_after is not None:
                        if retry_delay_seconds is None:
                            retry_delay_seconds = retry_after
                        else:
                            retry_delay_seconds = min(retry_delay_seconds, retry_after)
                    logger.warning(
                        "Gemini quota exhausted for model=%s: %s",
                        model_name,
                        trim_text_for_log(error_message),
                    )
                    continue

                logger.exception("Gemini request failed for model=%s", model_name)
                continue

        return None

    first_attempt = try_models()
    if first_attempt is not None:
        return first_attempt

    if saw_quota_exhaustion and retry_delay_seconds is not None:
        bounded_delay = max(1.0, min(retry_delay_seconds, 12.0))
        logger.info("Retrying Gemini request after %.2fs due to quota throttling", bounded_delay)
        time.sleep(bounded_delay)
        second_attempt = try_models()
        if second_attempt is not None:
            return second_attempt

    if saw_quota_exhaustion:
        return {
            "status": "error",
            "data": None,
            "error": (
                "Gemini quota exceeded for configured models. "
                "Set GEMINI_MODEL_CANDIDATES or check billing/limits."
            ),
        }

    return {
        "status": "error",
        "data": None,
        "error": f"Gemini request failed: {last_error_message or 'Unknown error'}",
    }
