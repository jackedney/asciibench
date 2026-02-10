"""Configuration service for unified configuration access with validation.

This module provides a ConfigService class that encapsulates loading and
validation of all configuration files (config.yaml, models.yaml, prompts.yaml,
evaluator_config.yaml) with Pydantic models for type safety.

The ConfigService uses a singleton pattern with lazy loading, ensuring that
configuration is loaded once and cached. Configuration is validated on load
with Pydantic models, providing clear error messages for invalid configurations.

Configuration schema:
    - config.yaml: Generation settings (temperature, tokens, attempts, etc.)
    - models.yaml: List of model configurations (id, name)
    - prompts.yaml: Prompt templates and word lists for expansion
    - evaluator_config.yaml: VLM evaluation settings (models, thresholds, concurrency)

Example:
    >>> # Get models (lazy loads from models.yaml)
    >>> models = ConfigService().get_models()
    >>> print(f"Loaded {len(models)} models")
    >>>
    >>> # Get prompts (lazy loads from prompts.yaml)
    >>> prompts = ConfigService().get_prompts()
    >>> print(f"Loaded {len(prompts)} prompts")
    >>>
    >>> # Get evaluator config (lazy loads from evaluator_config.yaml)
    >>> evaluator_config = ConfigService().get_evaluator_config()
    >>> print(f"VLM models: {evaluator_config.vlm_models}")
    >>>
    >>> # Get app config (lazy loads from config.yaml)
    >>> app_config = ConfigService().get_app_config()
    >>> print(f"Temperature: {app_config.temperature}")

Validation:
    All configuration is validated using Pydantic models on load. Invalid
    configuration raises pydantic.ValidationError with specific field information.

Singleton pattern:
    The ConfigService is a singleton with lazy loading. Multiple calls to
    ConfigService() return the same instance, and configuration is loaded
    once on first access.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from asciibench.common.config import EvaluatorConfig, GenerationConfig, TournamentConfig
from asciibench.common.models import Model, Prompt

# Note: dataclasses.field is used for ConfigCache (a dataclass),
# while pydantic.Field is used for Pydantic BaseModel classes

logger = logging.getLogger(__name__)


class ConfigServiceError(Exception):
    """Exception raised for configuration service errors."""

    pass


class TemplateDefinition(BaseModel):
    """Pydantic model for a template definition from prompts.yaml.

    Attributes:
        category: Category identifier for the template
        template: Template string with placeholders like {object}, {animal}
        word_list: Name of word list to use for single-variable templates
        word_list_pairs: List of explicit word pairs for multi-variable templates
    """

    category: str = "unknown"
    template: str = ""
    word_list: str | None = None
    word_list_pairs: list[dict[str, Any]] = Field(default_factory=list)


class WordLists(BaseModel):
    """Pydantic model for word lists from prompts.yaml.

    Attributes:
        objects: List of object names for templates
        animals: List of animal names for templates
        actions: List of action verbs for templates
        positions: List of positional words for templates
    """

    objects: list[str] = Field(default_factory=list)
    animals: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)


class PromptsConfig(BaseModel):
    """Pydantic model for full prompts.yaml structure.

    Supports two formats:
    1. Legacy format with 'prompts' key containing static prompts
    2. Template format with 'templates' and 'word_lists' keys

    Attributes:
        prompts: Legacy format list of static prompts
        templates: List of template definitions for expansion
        word_lists: Word lists for template expansion
    """

    prompts: list[dict[str, Any]] | None = None
    templates: list[TemplateDefinition] = Field(default_factory=list)
    word_lists: WordLists = Field(default_factory=WordLists)


class ModelsConfig(BaseModel):
    """Pydantic model for models.yaml structure.

    Attributes:
        models: List of model configurations
    """

    models: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class ConfigCache:
    """Cache for loaded configuration data.

    Attributes:
        models: Cached list of Model objects
        prompts: Cached list of Prompt objects
        evaluator_config: Cached EvaluatorConfig
        app_config: Cached GenerationConfig
        tournament_config: Cached TournamentConfig
        models_loaded: Flag indicating if models have been loaded
        models_loaded_path: Path from which models were loaded
        prompts_loaded: Flag indicating if prompts have been loaded
        prompts_loaded_path: Path from which prompts were loaded
        evaluator_config_loaded: Flag indicating if evaluator config has been loaded
        evaluator_config_loaded_path: Path from which evaluator config was loaded
        app_config_loaded: Flag indicating if app config has been loaded
        app_config_loaded_path: Path from which app config was loaded
        tournament_config_loaded: Flag indicating if tournament config has been loaded
        tournament_config_loaded_path: Path from which tournament config was loaded
    """

    models: list[Model] = field(default_factory=list)
    prompts: list[Prompt] = field(default_factory=list)
    evaluator_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    app_config: GenerationConfig = field(default_factory=GenerationConfig)
    tournament_config: TournamentConfig = field(default_factory=TournamentConfig)
    models_loaded: bool = False
    models_loaded_path: str | None = None
    prompts_loaded: bool = False
    prompts_loaded_path: str | None = None
    evaluator_config_loaded: bool = False
    evaluator_config_loaded_path: str | None = None
    app_config_loaded: bool = False
    app_config_loaded_path: str | None = None
    tournament_config_loaded: bool = False
    tournament_config_loaded_path: str | None = None


class ConfigService:
    """Singleton service for unified configuration access with validation.

    This service provides a single point of access for loading and validating
    all configuration files. It uses a singleton pattern with lazy loading,
    ensuring configuration is loaded once and cached.

    Thread safety:
        The singleton initialization is thread-safe using a double-checked
        locking pattern.

    Example:
        >>> # Get singleton instance
        >>> config = ConfigService()
        >>>
        >>> # Access configuration (lazy loaded on first access)
        >>> models = config.get_models()
        >>> prompts = config.get_prompts()
        >>> evaluator_config = config.get_evaluator_config()
        >>> app_config = config.get_app_config()
        >>>
        >>> # All subsequent calls return cached data
        >>> models_again = config.get_models()
        >>> assert models is models_again

    Raises:
        ConfigServiceError: If configuration files are not found or invalid
        ValidationError: If configuration structure is invalid (from Pydantic)
    """

    _instance: "ConfigService | None" = None
    _lock: Lock = Lock()
    _cache: ConfigCache

    def __new__(cls) -> "ConfigService":
        """Create or return singleton instance with thread-safe initialization.

        Returns:
            The singleton ConfigService instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._cache = ConfigCache()
                    cls._instance = instance
        # At this point _instance is guaranteed to be non-None
        assert cls._instance is not None
        return cls._instance

    def __init__(self) -> None:
        """Initialize ConfigService (noop for singleton)."""
        pass

    def _load_yaml_config(
        self,
        path: str,
        cache_field: str,
        file_name: str,
        parser_fn: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Generic helper for loading YAML configuration files with caching.

        Args:
            path: Path to the YAML file
            cache_field: Name of the cache field (e.g., 'models', 'prompts', etc.)
            file_name: Name of the file for error messages (e.g., 'models.yaml')
            parser_fn: Function to parse the loaded data into the final config object

        Returns:
            The parsed and cached configuration object.

        Raises:
            ConfigServiceError: If the file is not found or invalid
        """
        loaded_flag = f"{cache_field}_loaded"
        loaded_path = f"{cache_field}_loaded_path"

        if not getattr(self._cache, loaded_flag) or getattr(self._cache, loaded_path) != path:
            try:
                with Path(path).open() as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    data = {}
                parsed_value = parser_fn(data)
                setattr(self._cache, cache_field, parsed_value)
                setattr(self._cache, loaded_flag, True)
                setattr(self._cache, loaded_path, path)
                logger.debug(f"Loaded {file_name} from {path}")
            except FileNotFoundError as e:
                raise ConfigServiceError(f"{file_name} not found: {path}") from e
            except ValidationError as e:
                raise ConfigServiceError(f"Invalid {file_name} structure: {e}") from e

        return getattr(self._cache, cache_field)

    def get_models(self, path: str = "models.yaml") -> list[Model]:
        """Get validated list of model configurations.

        Loads models from the specified YAML file on first access,
        then returns cached data on subsequent calls with the same path.

        Args:
            path: Path to models.yaml file (default: "models.yaml")

        Returns:
            List of validated Model objects.

        Raises:
            ConfigServiceError: If models.yaml is not found
            ValidationError: If models.yaml structure is invalid

        Example:
            >>> config = ConfigService()
            >>> models = config.get_models()
            >>> for model in models:
            ...     print(f"{model.name}: {model.id}")
        """

        def parse_models(data: dict[str, Any]) -> list[Model]:
            models_config = ModelsConfig(**data)
            return [Model(**model_dict) for model_dict in models_config.models]

        return self._load_yaml_config(path, "models", "models.yaml", parse_models)

    def get_prompts(self, path: str = "prompts.yaml") -> list[Prompt]:
        """Get validated list of prompts (expanded from templates).

        Loads prompts from the specified YAML file on first access,
        expands templates using word lists, then returns cached data
        on subsequent calls with the same path.

        Args:
            path: Path to prompts.yaml file (default: "prompts.yaml")

        Returns:
            List of validated Prompt objects.

        Raises:
            ConfigServiceError: If prompts.yaml is not found
            ValidationError: If prompts.yaml structure is invalid

        Example:
            >>> config = ConfigService()
            >>> prompts = config.get_prompts()
            >>> for prompt in prompts:
            ...     print(f"[{prompt.category}] {prompt.text}")
        """

        def parse_prompts(data: dict[str, Any]) -> list[Prompt]:
            prompts_config = PromptsConfig(**data)
            if prompts_config.prompts is not None:
                return [Prompt(**prompt_dict) for prompt_dict in prompts_config.prompts]
            else:
                return self._expand_templates(prompts_config.templates, prompts_config.word_lists)

        return self._load_yaml_config(path, "prompts", "prompts.yaml", parse_prompts)

    def _expand_templates(
        self, templates: list[TemplateDefinition], word_lists: WordLists
    ) -> list[Prompt]:
        """Expand prompt templates using word lists.

        Supports both single-variable templates (using word_list) and
        multi-variable templates (using word_list_pairs).

        Args:
            templates: List of template definitions
            word_lists: Word lists for template expansion

        Returns:
            List of expanded Prompt objects.
        """
        prompts: list[Prompt] = []

        for template_def in templates:
            category = template_def.category
            template = template_def.template

            if template_def.word_list is not None:
                # Single word list expansion
                word_list_name = template_def.word_list
                words = getattr(word_lists, word_list_name, [])
                if not words:
                    continue

                placeholder = self._extract_placeholder(template)
                if placeholder:
                    for word in words:
                        text = template.replace(f"{{{placeholder}}}", word)
                        prompts.append(
                            Prompt(text=text, category=category, template_type="template")
                        )

            elif template_def.word_list_pairs:
                # Explicit pairs expansion (for multi-variable templates)
                pairs = template_def.word_list_pairs
                if not pairs:
                    continue

                for pair in pairs:
                    text = template
                    for key, value in pair.items():
                        text = text.replace(f"{{{key}}}", str(value))
                    prompts.append(Prompt(text=text, category=category, template_type="template"))

        return prompts

    @staticmethod
    def _extract_placeholder(template: str) -> str | None:
        """Extract the first placeholder name from a template string.

        Args:
            template: Template string with placeholders like {object}

        Returns:
            Placeholder name without braces, or None if not found.

        Example:
            >>> ConfigService._extract_placeholder("Draw a {object}")
            'object'
            >>> ConfigService._extract_placeholder("No placeholders here")
            None
        """
        start = template.find("{")
        end = template.find("}")
        if start != -1 and end != -1 and end > start:
            return template[start + 1 : end]
        return None

    def get_evaluator_config(self, path: str = "evaluator_config.yaml") -> EvaluatorConfig:
        """Get validated evaluator configuration.

        Loads evaluator configuration from the specified YAML file on first
        access, then returns cached data on subsequent calls with the same path.

        Args:
            path: Path to evaluator_config.yaml file (default: "evaluator_config.yaml")

        Returns:
            Validated EvaluatorConfig object.

        Raises:
            ConfigServiceError: If evaluator_config.yaml is not found
            ValidationError: If evaluator_config.yaml structure is invalid

        Example:
            >>> config = ConfigService()
            >>> eval_config = config.get_evaluator_config()
            >>> print(f"VLM models: {eval_config.vlm_models}")
            >>> print(f"Threshold: {eval_config.similarity_threshold}")
        """

        def parse_evaluator_config(data: dict[str, Any]) -> EvaluatorConfig:
            evaluator_data = data.get("evaluator", {})
            return EvaluatorConfig(**evaluator_data)

        return self._load_yaml_config(
            path, "evaluator_config", "evaluator_config.yaml", parse_evaluator_config
        )

    def get_app_config(self, path: str = "config.yaml") -> GenerationConfig:
        """Get validated application/generation configuration.

        Loads application configuration from the specified YAML file on first
        access, then returns cached data on subsequent calls with the same path.

        Args:
            path: Path to config.yaml file (default: "config.yaml")

        Returns:
            Validated GenerationConfig object.

        Raises:
            ConfigServiceError: If config.yaml is not found
            ValidationError: If config.yaml structure is invalid

        Example:
            >>> config = ConfigService()
            >>> app_config = config.get_app_config()
            >>> print(f"Temperature: {app_config.temperature}")
            >>> print(f"Max tokens: {app_config.max_tokens}")
            >>> print(f"Attempts per prompt: {app_config.attempts_per_prompt}")
        """

        def parse_app_config(data: dict[str, Any]) -> GenerationConfig:
            generation_data = data.get("generation", {})
            return GenerationConfig(**generation_data)

        return self._load_yaml_config(path, "app_config", "config.yaml", parse_app_config)

    def get_tournament_config(self, path: str = "config.yaml") -> TournamentConfig:
        """Get validated tournament configuration.

        Loads tournament configuration from the specified YAML file on first
        access, then returns cached data on subsequent calls with the same path.

        Args:
            path: Path to config.yaml file (default: "config.yaml")

        Returns:
            Validated TournamentConfig object.

        Raises:
            ConfigServiceError: If config.yaml is not found
            ValidationError: If config.yaml structure is invalid

        Example:
            >>> config = ConfigService()
            >>> tournament_config = config.get_tournament_config()
            >>> print(f"Round size: {tournament_config.round_size}")
        """

        def parse_tournament_config(data: dict[str, Any]) -> TournamentConfig:
            tournament_data = data.get("tournament", {})
            return TournamentConfig(**tournament_data)

        return self._load_yaml_config(
            path, "tournament_config", "config.yaml", parse_tournament_config
        )

    def clear_cache(self) -> None:
        """Clear all cached configuration data.

        Forces next access to reload configuration from files. This is useful
        when configuration files may have been modified externally.

        Example:
            >>> config = ConfigService()
            >>> models = config.get_models()  # Cached
            >>> # External process modifies models.yaml
            >>> config.clear_cache()
            >>> models = config.get_models()  # Reloads from file
        """
        self._cache = ConfigCache()
        logger.debug("Configuration cache cleared")
