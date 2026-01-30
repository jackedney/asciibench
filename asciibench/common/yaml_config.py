import yaml

from asciibench.common.models import Model, Prompt


def load_models(path: str = "models.yaml") -> list[Model]:
    with open(path) as f:
        data = yaml.safe_load(f)
    models_data = data.get("models", [])
    return [Model(**model) for model in models_data]


def load_prompts(path: str = "prompts.yaml") -> list[Prompt]:
    with open(path) as f:
        data = yaml.safe_load(f)
    prompts_data = data.get("prompts", [])
    return [Prompt(**prompt) for prompt in prompts_data]
