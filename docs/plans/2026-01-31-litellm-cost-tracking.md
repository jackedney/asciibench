# Plan: Use LiteLLM via smolagents for Cost Tracking

## Problem

Costs show as `$0.000000` because:
1. Current code uses `smolagents.OpenAIModel` which doesn't expose OpenRouter's cost data
2. smolagents' `LiteLLMModel` extracts tokens but **ignores** cost data from `response._hidden_params["response_cost"]`

## Solution

Replace `OpenAIModel` with `LiteLLMModel` and extend it to extract cost from LiteLLM's response.

## Implementation

### Step 1: Update `pyproject.toml`

Change dependency from `smolagents[openai]` to `smolagents[litellm]`:

```toml
# Before
"smolagents[openai]>=1.24.0",

# After
"smolagents[litellm]>=1.24.0",
```

### Step 2: Update `asciibench/generator/client.py`

**Key changes:**
1. Import `LiteLLMModel` instead of `OpenAIModel`
2. Create custom `LiteLLMModelWithCost` that extracts cost from `response._hidden_params["response_cost"]`
3. Auto-prepend `openrouter/` prefix to model IDs (keep models.yaml unchanged)
4. Remove the now-unnecessary custom `OpenRouterModel` class

**New implementation:**

```python
from smolagents import ChatMessage, LiteLLMModel

class LiteLLMModelWithCost(LiteLLMModel):
    """LiteLLMModel that extracts cost from response."""

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        # Call parent to get ChatMessage
        chat_message = super().generate(
            messages, stop_sequences, response_format, tools_to_call_from, **kwargs
        )

        # Extract cost from raw litellm response
        if chat_message.raw is not None:
            cost = None
            if hasattr(chat_message.raw, "_hidden_params"):
                cost = chat_message.raw._hidden_params.get("response_cost")
            if cost is not None:
                chat_message.raw._litellm_cost = cost

        return chat_message
```

**In `OpenRouterClient.generate()`:**
- Use `LiteLLMModelWithCost` instead of `OpenRouterModel`
- Auto-prepend `openrouter/` to model_id for LiteLLM (models.yaml stays unchanged)
- Extract cost from `raw._litellm_cost`

### Step 3: Update `tests/test_client.py`

- Change mock target from `OpenRouterModel` to `LiteLLMModelWithCost`
- Update mock response structure to include `_hidden_params` with `response_cost`

### Step 4: Run `uv sync` and verify

```bash
uv sync
uv run pytest tests/test_client.py -v
uv run python -m asciibench.generator.demo  # Manual test
```

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Change `smolagents[openai]` → `smolagents[litellm]` |
| `asciibench/generator/client.py` | Replace OpenAIModel with LiteLLMModel + cost extraction |
| `tests/test_client.py` | Update mocks for new class |

## Verification

1. Run tests: `uv run pytest tests/test_client.py -v`
2. Run demo: `uv run python -m asciibench.generator.demo`
3. Confirm cost shows non-zero values like `$0.001234`
