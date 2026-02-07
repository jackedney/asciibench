import yaml

from asciibench.common.config import EvaluatorConfig, GenerationConfig
from asciibench.common.models import Model, Prompt


def load_generation_config(path: str = "config.yaml") -> GenerationConfig:
    """Load generation configuration from a YAML file.

    Args:
        path: Path to the config YAML file

    Returns:
        GenerationConfig loaded from the file, or defaults if file doesn't exist
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return GenerationConfig()
        generation_data = data.get("generation", {})
        return GenerationConfig(**generation_data)
    except FileNotFoundError:
        return GenerationConfig()


def load_models(path: str = "models.yaml") -> list[Model]:
    with open(path) as f:
        data = yaml.safe_load(f)
    models_data = data.get("models", [])
    return [Model(**model) for model in models_data]


def load_prompts(path: str = "prompts.yaml") -> list[Prompt]:
    """Load and expand prompt templates into concrete prompts.

    Supports two YAML formats:
    1. Legacy format with 'prompts' key containing static prompts
    2. Template format with 'templates' and 'word_lists' keys

    For templates:
    - 'word_list': Uses a single word list to expand the template
    - 'word_list_pairs': Uses explicit pairs for multi-variable templates
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Legacy format: direct prompts list
    if "prompts" in data:
        prompts_data = data.get("prompts", [])
        return [Prompt(**prompt) for prompt in prompts_data]

    # Template format: expand templates using word lists
    word_lists = data.get("word_lists", {})
    templates = data.get("templates", [])
    prompts: list[Prompt] = []

    for template_def in templates:
        category = template_def.get("category", "unknown")
        template = template_def.get("template", "")

        if "word_list" in template_def:
            # Single word list expansion
            word_list_name = template_def["word_list"]
            words = word_lists.get(word_list_name, [])
            if not words:
                continue  # Empty word list produces no prompts

            # Determine the placeholder name from template
            placeholder = _extract_placeholder(template)
            if placeholder:
                for word in words:
                    text = template.replace(f"{{{placeholder}}}", word)
                    prompts.append(Prompt(text=text, category=category, template_type="template"))

        elif "word_list_pairs" in template_def:
            # Explicit pairs expansion (for multi-variable templates)
            pairs = template_def.get("word_list_pairs", [])
            if not pairs:
                continue  # Empty pairs list produces no prompts

            for pair in pairs:
                text = template
                for key, value in pair.items():
                    text = text.replace(f"{{{key}}}", str(value))
                prompts.append(Prompt(text=text, category=category, template_type="template"))

    return prompts


def _extract_placeholder(template: str) -> str | None:
    """Extract the first placeholder name from a template string.

    Example: 'Draw a {object} in ASCII' -> 'object'
    """
    start = template.find("{")
    end = template.find("}")
    if start != -1 and end != -1 and end > start:
        return template[start + 1 : end]
    return None


def load_evaluator_config(path: str = "evaluator_config.yaml") -> EvaluatorConfig:
    """Load evaluator configuration from a YAML file.

    Args:
        path: Path to the evaluator config YAML file

    Returns:
        EvaluatorConfig loaded from the file

    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return EvaluatorConfig()
        evaluator_data = data.get("evaluator", {})
        return EvaluatorConfig(**evaluator_data)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{path!r} not found. Please create an evaluator configuration file."
        ) from e
