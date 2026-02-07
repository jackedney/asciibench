import logging

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class LogfireConfig(BaseModel):
    token: str | None = None
    service_name: str = "asciibench"
    environment: str = "development"

    @property
    def is_enabled(self) -> bool:
        return bool(self.token)


class GenerationConfig(BaseModel):
    attempts_per_prompt: int = 5
    temperature: float = 0.0
    max_tokens: int = 1000
    provider: str = "openrouter"
    system_prompt: str = ""
    reasoning_effort: str | None = None  # For native OpenAI models
    reasoning: bool = False  # For OpenRouter reasoning models (o1, deepseek, kimi-k2)
    include_reasoning: bool = False  # Whether to include reasoning tokens in response
    max_concurrent_requests: int = 10

    @field_validator("max_concurrent_requests")
    @classmethod
    def validate_max_concurrent_requests(cls, v) -> int:
        """Validate max_concurrent_requests is positive."""
        if v <= 0:
            raise ValueError("max_concurrent_requests must be greater than 0")
        return v


class FontConfig(BaseModel):
    family: str = "Courier"
    size: int = 14

    @field_validator("size")
    @classmethod
    def validate_size(cls, v) -> int:
        """Validate font size is positive."""
        if v <= 0:
            raise ValueError("font size must be greater than 0")
        return v


class EvaluatorConfig(BaseModel):
    vlm_models: list[str] = []
    similarity_threshold: float = 0.7
    max_concurrency: int = 5
    font: FontConfig = Field(default_factory=FontConfig)

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v) -> float:
        """Validate similarity_threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return v

    @field_validator("max_concurrency")
    @classmethod
    def validate_max_concurrency(cls, v) -> int:
        """Validate max_concurrency is positive."""
        if v <= 0:
            raise ValueError("max_concurrency must be greater than 0")
        return v


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: int = 120
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logfire: LogfireConfig = Field(default_factory=LogfireConfig)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def timeout_seconds(self) -> int:
        """Alias for openrouter_timeout_seconds for backward compatibility."""
        return self.openrouter_timeout_seconds

    @field_validator("openrouter_timeout_seconds", mode="before")
    @classmethod
    def validate_openrouter_timeout_seconds(cls, v) -> int:
        """Validate and return openrouter_timeout_seconds, using default if invalid."""
        default_timeout = 120
        logger.debug(f"validate_timeout_seconds called with v={v!r}, type={type(v)}")
        if v is None:
            logger.debug("v is None, returning default")
            return default_timeout

        try:
            timeout = int(v)
            if timeout <= 0:
                logger.warning(
                    f"Invalid timeout_seconds value: {v}. Using default: {default_timeout}s"
                )
                return default_timeout
            logger.debug(f"Returning validated timeout: {timeout}")
            return timeout
        except (ValueError, TypeError):
            logger.warning(f"Invalid timeout_seconds value: {v}. Using default: {default_timeout}s")
            return default_timeout
