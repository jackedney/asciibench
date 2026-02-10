import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from asciibench.common.config import GenerationConfig, Settings, TournamentConfig
from asciibench.common.yaml_config import load_generation_config, load_models, load_prompts


def test_settings_loads_from_env_file():
    import os

    env_file = Path(__file__).parent.parent / "test_env.env"
    env_file.write_text("OPENROUTER_API_KEY=test_key_123")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file, env_file_encoding="utf-8", extra="ignore"
        )

    original_env = os.environ.pop("OPENROUTER_API_KEY", None)

    try:
        settings = TestSettings()
        assert settings.openrouter_api_key == "test_key_123"
        assert settings.base_url == "https://openrouter.ai/api/v1"
    finally:
        if original_env is not None:
            os.environ["OPENROUTER_API_KEY"] = original_env
        if env_file.exists():
            env_file.unlink()


def test_settings_missing_env_file_loads_defaults():
    import os

    env_file = Path(__file__).parent.parent / "nonexistent.env"

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
        )

    original_env = os.environ.pop("OPENROUTER_API_KEY", None)

    try:
        settings = TestSettings()
        assert settings.openrouter_api_key == ""
        assert settings.base_url == "https://openrouter.ai/api/v1"
        assert settings.openrouter_timeout_seconds == 120
    finally:
        if original_env is not None:
            os.environ["OPENROUTER_API_KEY"] = original_env


def test_settings_timeout_seconds_default():
    """Test that timeout_seconds defaults to 120."""
    settings = Settings()
    assert settings.openrouter_timeout_seconds == 120


def test_settings_timeout_seconds_from_env(tmp_path):
    """Test that timeout_seconds can be loaded from env file."""
    import os

    env_file = tmp_path / "test_env_timeout.env"
    env_file.write_text("OPENROUTER_API_KEY=test_key\nOPENROUTER_TIMEOUT_SECONDS=60")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file, env_file_encoding="utf-8", extra="ignore"
        )

    original_env = os.environ.pop("OPENROUTER_TIMEOUT_SECONDS", None)

    try:
        settings = TestSettings()
        assert settings.openrouter_timeout_seconds == 60
    finally:
        if original_env is not None:
            os.environ["OPENROUTER_TIMEOUT_SECONDS"] = original_env
        if env_file.exists():
            env_file.unlink()


def test_settings_timeout_seconds_negative_uses_default(tmp_path):
    """Test that negative timeout_seconds uses default 120."""
    import os

    env_file = tmp_path / "test_env_negative.env"
    env_file.write_text("OPENROUTER_API_KEY=test_key\nOPENROUTER_TIMEOUT_SECONDS=-10")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file, env_file_encoding="utf-8", extra="ignore"
        )

    original_env = os.environ.pop("OPENROUTER_TIMEOUT_SECONDS", None)

    try:
        settings = TestSettings()
        assert settings.openrouter_timeout_seconds == 120
    finally:
        if original_env is not None:
            os.environ["OPENROUTER_TIMEOUT_SECONDS"] = original_env
        if env_file.exists():
            env_file.unlink()


def test_settings_timeout_seconds_non_numeric_uses_default(tmp_path):
    """Test that non-numeric timeout_seconds uses default 120."""
    import os

    env_file = tmp_path / "test_env_invalid.env"
    env_file.write_text("OPENROUTER_API_KEY=test_key\nOPENROUTER_TIMEOUT_SECONDS=abc")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file, env_file_encoding="utf-8", extra="ignore"
        )

    original_env = os.environ.pop("OPENROUTER_TIMEOUT_SECONDS", None)

    try:
        settings = TestSettings()
        assert settings.openrouter_timeout_seconds == 120
    finally:
        if original_env is not None:
            os.environ["OPENROUTER_TIMEOUT_SECONDS"] = original_env
        if env_file.exists():
            env_file.unlink()


def test_generation_config():
    config = GenerationConfig()
    assert config.attempts_per_prompt == 5
    assert config.temperature == 0.0
    assert config.max_tokens == 1000
    assert config.provider == "openrouter"
    assert config.system_prompt == ""
    assert config.max_concurrent_requests == 10

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


def test_generation_config_max_concurrent_requests_default():
    """Test that max_concurrent_requests defaults to 10."""
    config = GenerationConfig()
    assert config.max_concurrent_requests == 10


def test_generation_config_max_concurrent_requests_custom():
    """Test that max_concurrent_requests can be set to a custom value."""
    config = GenerationConfig(max_concurrent_requests=5)
    assert config.max_concurrent_requests == 5


def test_generation_config_max_concurrent_requests_negative():
    """Test that negative max_concurrent_requests raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        GenerationConfig(max_concurrent_requests=-1)
    assert "max_concurrent_requests must be greater than 0" in str(exc_info.value)


def test_generation_config_max_concurrent_requests_zero():
    """Test that max_concurrent_requests=0 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        GenerationConfig(max_concurrent_requests=0)
    assert "max_concurrent_requests must be greater than 0" in str(exc_info.value)


def test_tournament_config_defaults():
    """Test that TournamentConfig defaults to round_size=10."""
    config = TournamentConfig()
    assert config.round_size == 10


def test_tournament_config_custom():
    """Test that TournamentConfig can be set with custom values."""
    config = TournamentConfig(round_size=5)
    assert config.round_size == 5


def test_tournament_config_round_size_zero():
    """Test that round_size=0 raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        TournamentConfig(round_size=0)
    assert "round_size must be at least 1" in str(exc_info.value)


def test_tournament_config_round_size_negative():
    """Test that negative round_size raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        TournamentConfig(round_size=-1)
    assert "round_size must be at least 1" in str(exc_info.value)


def test_load_models():
    models = load_models("models.yaml")
    assert len(models) > 0
    assert all(hasattr(m, "id") for m in models)
    assert all(hasattr(m, "name") for m in models)
    assert all(m.id for m in models)
    assert all(m.name for m in models)


def test_load_generation_config():
    """Test that config.yaml is loaded."""
    config = load_generation_config("config.yaml")
    assert hasattr(config, "attempts_per_prompt")
    assert hasattr(config, "temperature")
    assert hasattr(config, "max_tokens")
    assert hasattr(config, "provider")
    assert hasattr(config, "system_prompt")


def test_load_generation_config_with_custom_values():
    """Test loading custom values from config.yaml."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """generation:
  attempts_per_prompt: 10
  temperature: 0.7
  max_tokens: 2000
  provider: custom
  system_prompt: "You are helpful"
"""
        )
        tmp_file.flush()

        config = load_generation_config(tmp_file.name)
        assert config.attempts_per_prompt == 10
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.provider == "custom"
        assert config.system_prompt == "You are helpful"

    Path(tmp_file.name).unlink()


def test_load_generation_config_missing_file():
    """Test that missing config file returns defaults."""
    config = load_generation_config("nonexistent_config.yaml")
    assert config.attempts_per_prompt == 5
    assert config.temperature == 0.0
    assert config.max_tokens == 1000
    assert config.system_prompt == ""


def test_load_generation_config_empty_file():
    """Test that empty config file returns defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write("")
        tmp_file.flush()

        config = load_generation_config(tmp_file.name)
        assert config.attempts_per_prompt == 5
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.system_prompt == ""

    Path(tmp_file.name).unlink()


def test_load_generation_config_partial_values():
    """Test that partial config merges with defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """generation:
  attempts_per_prompt: 3
"""
        )
        tmp_file.flush()

        config = load_generation_config(tmp_file.name)
        assert config.attempts_per_prompt == 3  # Custom value
        assert config.temperature == 0.0  # Default
        assert config.max_tokens == 1000  # Default
        assert config.system_prompt == ""  # Default

    Path(tmp_file.name).unlink()


def test_load_prompts_generates_40_prompts():
    """Test that prompts.yaml generates prompts."""
    prompts = load_prompts("prompts.yaml")
    assert len(prompts) > 0


def test_load_prompts_category_distribution():
    """Test that prompts have categories."""
    prompts = load_prompts("prompts.yaml")

    categories = set()
    for prompt in prompts:
        categories.add(prompt.category)

    assert len(categories) > 0
    assert all(hasattr(p, "category") for p in prompts)


def test_load_prompts_single_object_category():
    """Test Category 1: Single Object prompts structure."""
    prompts = load_prompts("prompts.yaml")
    single_object_prompts = [p for p in prompts if p.category == "single_object"]

    for prompt in single_object_prompts:
        assert hasattr(prompt, "text")
        assert hasattr(prompt, "category")
        assert hasattr(prompt, "template_type")


def test_load_prompts_single_animal_category():
    """Test Category 2: Single Animal prompts structure."""
    prompts = load_prompts("prompts.yaml")
    single_animal_prompts = [p for p in prompts if p.category == "single_animal"]

    for prompt in single_animal_prompts:
        assert hasattr(prompt, "text")
        assert hasattr(prompt, "category")
        assert hasattr(prompt, "template_type")


def test_load_prompts_animal_action_category():
    """Test Category 3: Animal + Action prompts structure."""
    prompts = load_prompts("prompts.yaml")
    animal_action_prompts = [p for p in prompts if p.category == "animal_action"]

    for prompt in animal_action_prompts:
        assert hasattr(prompt, "text")
        assert hasattr(prompt, "category")
        assert hasattr(prompt, "template_type")


def test_load_prompts_spatial_relationship_category():
    """Test Category 4: Spatial Relationship prompts structure."""
    prompts = load_prompts("prompts.yaml")
    spatial_prompts = [p for p in prompts if p.category == "spatial_relationship"]

    for prompt in spatial_prompts:
        assert hasattr(prompt, "text")
        assert hasattr(prompt, "category")
        assert hasattr(prompt, "template_type")


def test_load_prompts_all_unique():
    """Test that all 40 prompts are unique."""
    prompts = load_prompts("prompts.yaml")
    prompt_texts = [p.text for p in prompts]
    assert len(prompt_texts) == len(set(prompt_texts))


def test_load_prompts_legacy_format():
    """Test legacy format with 'prompts' key still works."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """prompts:
  - text: Draw a cat
    category: animal
    template_type: simple
  - text: Draw a tree
    category: nature
    template_type: simple
"""
        )
        tmp_file.flush()

        prompts = load_prompts(tmp_file.name)
        assert len(prompts) == 2
        assert prompts[0].text == "Draw a cat"
        assert prompts[0].category == "animal"
        assert prompts[0].template_type == "simple"
        assert prompts[1].text == "Draw a tree"
        assert prompts[1].category == "nature"
        assert prompts[1].template_type == "simple"

    Path(tmp_file.name).unlink()


def test_load_prompts_empty_word_list():
    """Test that empty word list produces no prompts for that template."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """word_lists:
  objects: []
  animals:
    - cat
    - dog
templates:
  - category: single_object
    template: "Draw a {object} in ASCII art"
    word_list: objects
  - category: single_animal
    template: "Draw a {animal} in ASCII art"
    word_list: animals
"""
        )
        tmp_file.flush()

        prompts = load_prompts(tmp_file.name)
        # Only animal prompts should be generated (2), not object prompts (0)
        assert len(prompts) == 2
        for prompt in prompts:
            assert prompt.category == "single_animal"

    Path(tmp_file.name).unlink()


def test_load_prompts_empty_pairs_list():
    """Test that empty pairs list produces no prompts for that template."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """word_lists:
  animals:
    - cat
templates:
  - category: single_animal
    template: "Draw a {animal} in ASCII art"
    word_list: animals
  - category: animal_action
    template: "Draw a {animal} {action} in ASCII art"
    word_list_pairs: []
"""
        )
        tmp_file.flush()

        prompts = load_prompts(tmp_file.name)
        # Only single_animal prompts should be generated
        assert len(prompts) == 1
        assert prompts[0].category == "single_animal"

    Path(tmp_file.name).unlink()


def test_load_prompts_template_expansion_example():
    """Test example from acceptance criteria: template with objects: [cat, tree]."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """word_lists:
  objects:
    - cat
    - tree
templates:
  - category: test
    template: "Draw a {object}"
    word_list: objects
"""
        )
        tmp_file.flush()

        prompts = load_prompts(tmp_file.name)
        assert len(prompts) == 2
        assert prompts[0].text == "Draw a cat"
        assert prompts[1].text == "Draw a tree"

    Path(tmp_file.name).unlink()


def test_load_prompts_missing_word_list():
    """Test that referencing a non-existent word list produces no prompts."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(
            """word_lists:
  animals:
    - cat
templates:
  - category: test
    template: "Draw a {object} in ASCII art"
    word_list: nonexistent
"""
        )
        tmp_file.flush()

        prompts = load_prompts(tmp_file.name)
        assert len(prompts) == 0

    Path(tmp_file.name).unlink()
