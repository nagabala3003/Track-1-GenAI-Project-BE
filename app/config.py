import os
from dataclasses import dataclass
from typing import Final

ENVIRONMENT_ENV_VAR: Final[str] = "ENVIRONMENT"
PORT_ENV_VAR: Final[str] = "PORT"
GEMINI_API_KEY_ENV_VAR: Final[str] = "GEMINI_API_KEY"

ENV_DEV: Final[str] = "dev"
ENV_PROD: Final[str] = "prod"


@dataclass(frozen=True)
class Settings:
    environment: str
    host: str
    port: int
    reload: bool
    log_level: str
    gemini_api_key: str


def _resolve_environment() -> str:
    value = os.getenv(ENVIRONMENT_ENV_VAR, ENV_DEV).strip().lower()
    return value if value in {ENV_DEV, ENV_PROD} else ENV_DEV


def _resolve_port(environment: str) -> int:
    if environment == ENV_PROD:
        return int(os.getenv(PORT_ENV_VAR, "8080"))
    return 8000


def _resolve_log_level(environment: str) -> str:
    default_level = "DEBUG" if environment == ENV_DEV else "INFO"
    return os.getenv("LOG_LEVEL", default_level).strip().upper()


def load_settings() -> Settings:
    environment = _resolve_environment()
    return Settings(
        environment=environment,
        host="0.0.0.0",
        port=_resolve_port(environment),
        reload=environment == ENV_DEV,
        log_level=_resolve_log_level(environment),
        gemini_api_key=os.getenv(GEMINI_API_KEY_ENV_VAR, ""),
    )


settings = load_settings()
