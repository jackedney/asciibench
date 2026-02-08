"""Unit tests for ConfigService."""

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.config_service import (
    ConfigService,
    ConfigServiceError,
    PromptsConfig,
    TemplateDefinition,
    WordLists,
)
from asciibench.common.models import Model, Prompt


class TestConfigServiceSingleton:
    """Tests for ConfigService singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test that ConfigService returns same instance."""
        config1 = ConfigService()
        config2 = ConfigService()
        assert config1 is config2

    def test_singleton_thread_safe(self):
        """Test that ConfigService is thread-safe (basic test)."""
        import threading

        instances = []

        def get_instance():
            instances.append(ConfigService())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(instance is instances[0] for instance in instances)


class TestConfigServiceGetModels:
    """Tests for ConfigService.get_models() method."""

    def test_get_models_from_valid_file(self, tmp_path):
        """Test loading models from a valid models.yaml."""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(
            """models:
  - id: test/model-1
    name: Test Model 1
  - id: test/model-2
    name: Test Model 2
"""
        )

        config_service = ConfigService()
        config_service._cache.models_loaded = False
        models = config_service.get_models(path=str(config_file))

        assert len(models) == 2
        assert models[0].id == "test/model-1"
        assert models[0].name == "Test Model 1"
        assert models[1].id == "test/model-2"
        assert models[1].name == "Test Model 2"

    def test_get_models_caches_result(self, tmp_path):
        """Test that models are cached after first load."""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(
            """models:
  - id: test/model-1
    name: Test Model 1
"""
        )

        config_service = ConfigService()
        config_service._cache.models_loaded = False

        models1 = config_service.get_models(path=str(config_file))
        models2 = config_service.get_models(path=str(config_file))

        assert models1 is models2

    def test_get_models_file_not_found(self, tmp_path):
        """Test that FileNotFoundError raises ConfigServiceError."""
        config_service = ConfigService()
        config_service._cache.models_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_models(path=str(tmp_path / "nonexistent.yaml"))

        assert "not found" in str(exc_info.value)

    def test_get_models_invalid_yaml_structure(self, tmp_path):
        """Test that invalid YAML structure raises ConfigServiceError."""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(
            """models:
  - id: test/model-1
    # Missing 'name' field
"""
        )

        config_service = ConfigService()
        config_service._cache.models_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_models(path=str(config_file))

        assert "Invalid models.yaml structure" in str(exc_info.value)
        assert "validation error" in str(exc_info.value.__cause__).lower()


class TestConfigServiceGetPrompts:
    """Tests for ConfigService.get_prompts() method."""

    def test_get_prompts_from_legacy_format(self, tmp_path):
        """Test loading prompts from legacy format (direct prompts list)."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """prompts:
  - text: Draw a cat
    category: animal
    template_type: simple
  - text: Draw a tree
    category: nature
    template_type: simple
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False
        prompts = config_service.get_prompts(path=str(config_file))

        assert len(prompts) == 2
        assert prompts[0].text == "Draw a cat"
        assert prompts[0].category == "animal"
        assert prompts[1].text == "Draw a tree"
        assert prompts[1].category == "nature"

    def test_get_prompts_from_template_format(self, tmp_path):
        """Test loading prompts from template format with word lists."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """word_lists:
  objects:
    - cat
    - tree
templates:
  - category: single_object
    template: "Draw a {object}"
    word_list: objects
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False
        prompts = config_service.get_prompts(path=str(config_file))

        assert len(prompts) == 2
        assert prompts[0].text == "Draw a cat"
        assert prompts[0].category == "single_object"
        assert prompts[1].text == "Draw a tree"
        assert prompts[1].category == "single_object"

    def test_get_prompts_with_word_list_pairs(self, tmp_path):
        """Test template expansion with explicit word list pairs."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """templates:
  - category: animal_action
    template: "Draw a {animal} {action}"
    word_list_pairs:
      - animal: cat
        action: sleeping
      - animal: dog
        action: running
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False
        prompts = config_service.get_prompts(path=str(config_file))

        assert len(prompts) == 2
        assert prompts[0].text == "Draw a cat sleeping"
        assert prompts[1].text == "Draw a dog running"

    def test_get_prompts_caches_result(self, tmp_path):
        """Test that prompts are cached after first load."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """prompts:
  - text: Draw a cat
    category: animal
    template_type: simple
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False

        prompts1 = config_service.get_prompts(path=str(config_file))
        prompts2 = config_service.get_prompts(path=str(config_file))

        assert prompts1 is prompts2

    def test_get_prompts_file_not_found(self, tmp_path):
        """Test that FileNotFoundError raises ConfigServiceError."""
        config_service = ConfigService()
        config_service._cache.prompts_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_prompts(path=str(tmp_path / "nonexistent.yaml"))

        assert "not found" in str(exc_info.value)

    def test_get_prompts_invalid_yaml_structure(self, tmp_path):
        """Test that invalid YAML structure raises ConfigServiceError."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """prompts:
  - text: Draw a cat
    # Missing 'category' field
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_prompts(path=str(config_file))

        assert "Invalid prompts.yaml structure" in str(exc_info.value)

    def test_get_prompts_empty_word_list(self, tmp_path):
        """Test that empty word list produces no prompts for that template."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """word_lists:
  objects: []
  animals:
    - cat
templates:
  - category: single_object
    template: "Draw a {object}"
    word_list: objects
  - category: single_animal
    template: "Draw a {animal}"
    word_list: animals
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False
        prompts = config_service.get_prompts(path=str(config_file))

        assert len(prompts) == 1
        assert prompts[0].category == "single_animal"
        assert prompts[0].text == "Draw a cat"

    def test_get_prompts_empty_pairs_list(self, tmp_path):
        """Test that empty pairs list produces no prompts for that template."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """templates:
  - category: test_single
    template: "Draw a {animal}"
    word_list_pairs:
      - animal: cat
  - category: test_multi
    template: "Draw a {animal} {action}"
    word_list_pairs: []
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False
        prompts = config_service.get_prompts(path=str(config_file))

        assert len(prompts) == 1
        assert prompts[0].category == "test_single"


class TestConfigServiceGetEvaluatorConfig:
    """Tests for ConfigService.get_evaluator_config() method."""

    def test_get_evaluator_config_from_valid_file(self, tmp_path):
        """Test loading evaluator config from a valid evaluator_config.yaml."""
        config_file = tmp_path / "evaluator_config.yaml"
        config_file.write_text(
            """evaluator:
  vlm_models:
    - openai/gpt-4o
    - anthropic/claude-3.5-sonnet
  similarity_threshold: 0.8
  max_concurrency: 10
"""
        )

        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False
        eval_config = config_service.get_evaluator_config(path=str(config_file))

        assert eval_config.vlm_models == ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]
        assert eval_config.similarity_threshold == 0.8
        assert eval_config.max_concurrency == 10

    def test_get_evaluator_config_defaults(self, tmp_path):
        """Test that missing fields use defaults."""
        config_file = tmp_path / "evaluator_config.yaml"
        config_file.write_text(
            """evaluator:
  vlm_models:
    - test/model
"""
        )

        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False
        eval_config = config_service.get_evaluator_config(path=str(config_file))

        assert eval_config.similarity_threshold == 0.7  # Default value
        assert eval_config.max_concurrency == 5  # Default value

    def test_get_evaluator_config_caches_result(self, tmp_path):
        """Test that evaluator config is cached after first load."""
        config_file = tmp_path / "evaluator_config.yaml"
        config_file.write_text(
            """evaluator:
  vlm_models:
    - test/model
"""
        )

        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False

        config1 = config_service.get_evaluator_config(path=str(config_file))
        config2 = config_service.get_evaluator_config(path=str(config_file))

        assert config1 is config2

    def test_get_evaluator_config_file_not_found(self, tmp_path):
        """Test that FileNotFoundError raises ConfigServiceError."""
        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_evaluator_config(path=str(tmp_path / "nonexistent.yaml"))

        assert "not found" in str(exc_info.value)

    def test_get_evaluator_config_invalid_yaml_structure(self, tmp_path):
        """Test that invalid YAML structure raises ConfigServiceError."""
        config_file = tmp_path / "evaluator_config.yaml"
        config_file.write_text(
            """evaluator:
  similarity_threshold: invalid_value
"""
        )

        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_evaluator_config(path=str(config_file))

        assert "Invalid evaluator_config.yaml structure" in str(exc_info.value)

    def test_get_evaluator_config_invalid_similarity_threshold(self, tmp_path):
        """Test that invalid similarity_threshold raises ConfigServiceError."""
        config_file = tmp_path / "evaluator_config.yaml"
        config_file.write_text(
            """evaluator:
  similarity_threshold: 1.5
"""
        )

        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_evaluator_config(path=str(config_file))

        assert "Invalid evaluator_config.yaml structure" in str(exc_info.value)


class TestConfigServiceGetAppConfig:
    """Tests for ConfigService.get_app_config() method."""

    def test_get_app_config_from_valid_file(self, tmp_path):
        """Test loading app config from a valid config.yaml."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """generation:
  attempts_per_prompt: 10
  temperature: 0.7
  max_tokens: 2000
  provider: custom
  system_prompt: "You are helpful"
"""
        )

        config_service = ConfigService()
        config_service._cache.app_config_loaded = False
        app_config = config_service.get_app_config(path=str(config_file))

        assert app_config.attempts_per_prompt == 10
        assert app_config.temperature == 0.7
        assert app_config.max_tokens == 2000
        assert app_config.provider == "custom"
        assert app_config.system_prompt == "You are helpful"

    def test_get_app_config_defaults(self, tmp_path):
        """Test that missing fields use defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """generation:
  attempts_per_prompt: 3
"""
        )

        config_service = ConfigService()
        config_service._cache.app_config_loaded = False
        app_config = config_service.get_app_config(path=str(config_file))

        assert app_config.attempts_per_prompt == 3  # Custom value
        assert app_config.temperature == 0.0  # Default
        assert app_config.max_tokens == 1000  # Default
        assert app_config.provider == "openrouter"  # Default

    def test_get_app_config_caches_result(self, tmp_path):
        """Test that app config is cached after first load."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """generation:
  attempts_per_prompt: 5
"""
        )

        config_service = ConfigService()
        config_service._cache.app_config_loaded = False

        config1 = config_service.get_app_config(path=str(config_file))
        config2 = config_service.get_app_config(path=str(config_file))

        assert config1 is config2

    def test_get_app_config_file_not_found(self, tmp_path):
        """Test that FileNotFoundError raises ConfigServiceError."""
        config_service = ConfigService()
        config_service._cache.app_config_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_app_config(path=str(tmp_path / "nonexistent.yaml"))

        assert "not found" in str(exc_info.value)

    def test_get_app_config_invalid_yaml_structure(self, tmp_path):
        """Test that invalid YAML structure raises ConfigServiceError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """generation:
  attempts_per_prompt: invalid_value
"""
        )

        config_service = ConfigService()
        config_service._cache.app_config_loaded = False

        with pytest.raises(ConfigServiceError) as exc_info:
            config_service.get_app_config(path=str(config_file))

        assert "Invalid config.yaml structure" in str(exc_info.value)


class TestConfigServiceClearCache:
    """Tests for ConfigService.clear_cache() method."""

    def test_clear_cache_clears_models(self, tmp_path):
        """Test that clear_cache clears models cache."""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(
            """models:
  - id: test/model-1
    name: Test Model 1
"""
        )

        config_service = ConfigService()
        config_service._cache.models_loaded = False

        models1 = config_service.get_models(path=str(config_file))
        config_service.clear_cache()
        models2 = config_service.get_models(path=str(config_file))

        assert models1 is not models2

    def test_clear_cache_clears_prompts(self, tmp_path):
        """Test that clear_cache clears prompts cache."""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(
            """prompts:
  - text: Draw a cat
    category: animal
    template_type: simple
"""
        )

        config_service = ConfigService()
        config_service._cache.prompts_loaded = False

        prompts1 = config_service.get_prompts(path=str(config_file))
        config_service.clear_cache()
        prompts2 = config_service.get_prompts(path=str(config_file))

        assert prompts1 is not prompts2

    def test_clear_cache_clears_evaluator_config(self, tmp_path):
        """Test that clear_cache clears evaluator config cache."""
        config_file = tmp_path / "evaluator_config.yaml"
        config_file.write_text(
            """evaluator:
  vlm_models:
    - test/model
"""
        )

        config_service = ConfigService()
        config_service._cache.evaluator_config_loaded = False

        config1 = config_service.get_evaluator_config(path=str(config_file))
        config_service.clear_cache()
        config2 = config_service.get_evaluator_config(path=str(config_file))

        assert config1 is not config2

    def test_clear_cache_clears_app_config(self, tmp_path):
        """Test that clear_cache clears app config cache."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """generation:
  attempts_per_prompt: 5
"""
        )

        config_service = ConfigService()
        config_service._cache.app_config_loaded = False

        config1 = config_service.get_app_config(path=str(config_file))
        config_service.clear_cache()
        config2 = config_service.get_app_config(path=str(config_file))

        assert config1 is not config2


class TestExtractPlaceholder:
    """Tests for ConfigService._extract_placeholder() static method."""

    def test_extract_single_placeholder(self):
        """Test extracting a single placeholder."""
        assert ConfigService._extract_placeholder("Draw a {object}") == "object"
        assert ConfigService._extract_placeholder("Draw a {animal}") == "animal"
        assert ConfigService._extract_placeholder("{color} is my favorite") == "color"

    def test_extract_first_placeholder_only(self):
        """Test that only first placeholder is extracted."""
        assert ConfigService._extract_placeholder("Draw a {animal} {action}") == "animal"

    def test_extract_placeholder_with_spaces(self):
        """Test extracting placeholder with spaces around."""
        assert ConfigService._extract_placeholder("Draw a { object }") == " object "

    def test_extract_placeholder_none(self):
        """Test that None is returned when no placeholder found."""
        assert ConfigService._extract_placeholder("Draw a cat") is None
        assert ConfigService._extract_placeholder("") is None
        assert ConfigService._extract_placeholder("{") is None
        assert ConfigService._extract_placeholder("}") is None


class TestConfigModels:
    """Tests for Pydantic config models."""

    def test_template_definition_valid(self):
        """Test TemplateDefinition with valid data."""
        template = TemplateDefinition(
            category="test", template="Draw a {object}", word_list="objects"
        )
        assert template.category == "test"
        assert template.template == "Draw a {object}"
        assert template.word_list == "objects"

    def test_template_definition_defaults(self):
        """Test TemplateDefinition default values."""
        template = TemplateDefinition()
        assert template.category == "unknown"
        assert template.template == ""
        assert template.word_list is None
        assert template.word_list_pairs == []

    def test_word_lists_defaults(self):
        """Test WordLists default values."""
        word_lists = WordLists()
        assert word_lists.objects == []
        assert word_lists.animals == []
        assert word_lists.actions == []
        assert word_lists.positions == []

    def test_prompts_config_legacy_format(self):
        """Test PromptsConfig with legacy format."""
        config = PromptsConfig(
            prompts=[{"text": "Draw a cat", "category": "animal", "template_type": "simple"}]
        )
        assert config.prompts is not None
        assert len(config.prompts) == 1
        assert config.templates == []

    def test_prompts_config_template_format(self):
        """Test PromptsConfig with template format."""
        config = PromptsConfig(
            templates=[
                TemplateDefinition(category="test", template="Draw a {object}", word_list="objects")
            ]
        )
        assert config.prompts is None
        assert len(config.templates) == 1


class TestConfigServiceIntegration:
    """Integration tests for ConfigService with real config files."""

    def test_load_from_actual_config_files(self):
        """Test loading from actual config files in repository."""
        from asciibench.common.config import EvaluatorConfig

        config_service = ConfigService()
        config_service.clear_cache()

        # Load models
        models = config_service.get_models()
        assert len(models) > 0
        assert all(isinstance(model, Model) for model in models)

        # Load prompts
        prompts = config_service.get_prompts()
        assert len(prompts) > 0
        assert all(isinstance(prompt, Prompt) for prompt in prompts)

        # Load evaluator config
        eval_config = config_service.get_evaluator_config()
        assert isinstance(eval_config, EvaluatorConfig)

        # Load app config
        app_config = config_service.get_app_config()
        assert isinstance(app_config, GenerationConfig)
