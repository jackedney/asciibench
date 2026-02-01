from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationConfig(BaseModel):
    attempts_per_prompt: int = 5
    temperature: float = 0.0
    max_tokens: int = 1000
    provider: str = "openrouter"
    system_prompt: str = ""
    reasoning_effort: str | None = None  # For native OpenAI models
    reasoning: bool = False  # For OpenRouter reasoning models (o1, deepseek, kimi-k2)
    include_reasoning: bool = False  # Whether to include reasoning tokens in response


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
