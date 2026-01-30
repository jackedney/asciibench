from pathlib import Path

from pydantic_settings import SettingsConfigDict

from asciibench.common.config import GenerationConfig, Settings
from asciibench.common.yaml_config import load_models, load_prompts


def test_settings_default_values():
    settings = Settings()
    assert settings.openrouter_api_key == ""
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.generation.attempts_per_prompt == 5
    assert settings.generation.temperature == 0.0
    assert settings.generation.max_tokens == 1000
    assert settings.generation.provider == "openrouter"
    assert settings.generation.system_prompt == ""


def test_settings_loads_from_env_file():
    env_file = Path(__file__).parent.parent / "test_env.env"
    env_file.write_text("OPENROUTER_API_KEY=test_key_123")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file, env_file_encoding="utf-8", extra="ignore"
        )

    try:
        settings = TestSettings()
        assert settings.openrouter_api_key == "test_key_123"
        assert settings.base_url == "https://openrouter.ai/api/v1"
    finally:
        env_file.unlink()


def test_settings_missing_env_file_loads_defaults():
    env_file = Path(__file__).parent.parent / "nonexistent.env"

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file, env_file_encoding="utf-8", extra="ignore"
        )

    settings = TestSettings()
    assert settings.openrouter_api_key == ""
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_generation_config():
    config = GenerationConfig()
    assert config.attempts_per_prompt == 5
    assert config.temperature == 0.0
    assert config.max_tokens == 1000
    assert config.provider == "openrouter"
    assert config.system_prompt == ""

    config_custom = GenerationConfig(
        attempts_per_prompt=10,
        temperature=0.7,
        max_tokens=2000,
        provider="custom",
        system_prompt="You are a helpful assistant.",
    )
    assert config_custom.attempts_per_prompt == 10
    assert config_custom.temperature == 0.7
    assert config_custom.max_tokens == 2000
    assert config_custom.provider == "custom"
    assert config_custom.system_prompt == "You are a helpful assistant."


def test_load_models():
    models = load_models("models.yaml")
    assert len(models) == 2
    assert models[0].id == "openai/gpt-4o"
    assert models[0].name == "GPT-4o"
    assert models[1].id == "anthropic/claude-3.5-sonnet"
    assert models[1].name == "Claude 3.5 Sonnet"


def test_load_prompts():
    prompts = load_prompts("prompts.yaml")
    assert len(prompts) == 2
    assert prompts[0].text == "Draw a cat"
    assert prompts[0].category == "animal"
    assert prompts[0].template_type == "simple"
    assert prompts[1].text == "Draw a tree"
    assert prompts[1].category == "nature"
    assert prompts[1].template_type == "simple"
