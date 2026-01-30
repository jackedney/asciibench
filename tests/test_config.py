import tempfile
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from asciibench.common.config import GenerationConfig, Settings
from asciibench.common.yaml_config import load_generation_config, load_models, load_prompts


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


def test_load_generation_config():
    """Test that config.yaml is loaded correctly."""
    config = load_generation_config("config.yaml")
    assert config.attempts_per_prompt == 5
    assert config.temperature == 0.0
    assert config.max_tokens == 1000
    assert config.provider == "openrouter"


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


def test_load_generation_config_empty_file():
    """Test that empty config file returns defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write("")
        tmp_file.flush()

        config = load_generation_config(tmp_file.name)
        assert config.attempts_per_prompt == 5
        assert config.temperature == 0.0
        assert config.max_tokens == 1000

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

    Path(tmp_file.name).unlink()


def test_load_prompts_generates_40_prompts():
    """Test that prompts.yaml expands to exactly 40 prompts."""
    prompts = load_prompts("prompts.yaml")
    assert len(prompts) == 40


def test_load_prompts_category_distribution():
    """Test that each category has exactly 10 prompts."""
    prompts = load_prompts("prompts.yaml")

    categories = {}
    for prompt in prompts:
        categories[prompt.category] = categories.get(prompt.category, 0) + 1

    assert categories.get("single_object", 0) == 10
    assert categories.get("single_animal", 0) == 10
    assert categories.get("animal_action", 0) == 10
    assert categories.get("spatial_relationship", 0) == 10


def test_load_prompts_single_object_category():
    """Test Category 1: Single Object prompts."""
    prompts = load_prompts("prompts.yaml")
    single_object_prompts = [p for p in prompts if p.category == "single_object"]

    assert len(single_object_prompts) == 10
    # Verify format: 'Draw a [OBJECT] in ASCII art'
    for prompt in single_object_prompts:
        assert prompt.text.startswith("Draw a ")
        assert prompt.text.endswith(" in ASCII art")
        assert prompt.template_type == "template"


def test_load_prompts_single_animal_category():
    """Test Category 2: Single Animal prompts."""
    prompts = load_prompts("prompts.yaml")
    single_animal_prompts = [p for p in prompts if p.category == "single_animal"]

    assert len(single_animal_prompts) == 10
    # Verify format: 'Draw a [ANIMAL] in ASCII art'
    for prompt in single_animal_prompts:
        assert prompt.text.startswith("Draw a ")
        assert prompt.text.endswith(" in ASCII art")
        assert prompt.template_type == "template"


def test_load_prompts_animal_action_category():
    """Test Category 3: Animal + Action prompts."""
    prompts = load_prompts("prompts.yaml")
    animal_action_prompts = [p for p in prompts if p.category == "animal_action"]

    assert len(animal_action_prompts) == 10
    # Verify format: 'Draw a [ANIMAL] [ACTION] in ASCII art'
    for prompt in animal_action_prompts:
        assert prompt.text.startswith("Draw a ")
        assert prompt.text.endswith(" in ASCII art")
        assert prompt.template_type == "template"


def test_load_prompts_spatial_relationship_category():
    """Test Category 4: Spatial Relationship prompts."""
    prompts = load_prompts("prompts.yaml")
    spatial_prompts = [p for p in prompts if p.category == "spatial_relationship"]

    assert len(spatial_prompts) == 10
    # Verify format: 'Draw a [OBJECT_A] [POSITION] a [OBJECT_B] in ASCII art'
    for prompt in spatial_prompts:
        assert prompt.text.startswith("Draw a ")
        assert prompt.text.endswith(" in ASCII art")
        assert " a " in prompt.text  # Contains spatial relationship
        assert prompt.template_type == "template"


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
