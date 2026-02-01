# Fix Failing Demo Models - Design Document

**Date:** 2026-02-01
**Status:** Implemented

## Problem

6-7 models out of 25 are failing in the demo script due to various issues:

| Model | Model ID | Failure Type | Details |
|-------|----------|--------------|---------|
| MiniMax M2.1 | `minimax/minimax-m2.1` | Repetitive output | 18,451 chars - kept repeating identical lines |
| GPT-4.1 Mini | `openai/gpt-4.1-mini` | Invalid code block | Extra spaces after backticks |
| GLM 4.5 Air | `z-ai/glm-4.5-air:free` | Timeout | Timed out after 120 seconds |
| Kimi K2 Thinking | `moonshotai/kimi-k2-thinking` | Empty output | Thinking consumed all tokens |
| Gemini 2.5 Flash Lite | `google/gemini-2.5-flash-lite` | Repetitive output | 44,888 chars - repeating lines |
| Trinity Large Preview | `arcee-ai/trinity-large-preview:free` | Timeout | Timed out after 120 seconds |
| Phi 4 | `microsoft/phi-4` | Timeout | Timed out after 120 seconds |

## Solution

Four-pronged approach addressing each failure category:

### 1. Reduce `max_tokens` globally (fixes repetitive outputs)

Change `config.yaml`:
```yaml
max_tokens: 2000  # Reduced from 10000
```

ASCII art typically needs 100-500 tokens. 10K tokens allows models to loop indefinitely.

### 2. Add `reasoning_effort` parameter (fixes thinking model exhaustion)

Add to `config.yaml`:
```yaml
reasoning_effort: low
```

LiteLLM maps this to provider-specific parameters:
- Anthropic: `thinking.budget_tokens`
- Gemini: `thinkingLevel`
- OpenAI: `reasoning_effort`

### 3. Relax sanitizer regex (fixes malformed code blocks)

Current pattern rejects ```` ```  ```` (spaces after backticks).

Change `sanitizer.py`:
```python
# Before
pattern = r"```(?:(?:text|ascii|plaintext)\s*\n|\n)(.*?)```"

# After - allow optional spaces after opening backticks
pattern = r"```\s*(?:(?:text|ascii|plaintext)\s*\n|\n)(.*?)```"
```

### 4. Timeouts

Lower token limits may help slow models finish faster. Models that still timeout may be infrastructure issues with those providers.

## Implementation Checklist

- [x] Update `config.yaml`: set `max_tokens: 2000`, add `reasoning_effort: low`
- [x] Update `asciibench/common/config.py`: add `reasoning_effort` field to `GenerationConfig`
- [x] Update `asciibench/generator/client.py`: pass `reasoning_effort` to LiteLLM
- [x] Update `asciibench/generator/sanitizer.py`: relax regex pattern
- [x] Add tests for whitespace tolerance in sanitizer (3 new tests)
- [ ] Clear existing results and re-run demo to verify fixes

## Expected Outcomes

| Failure Category | Fix | Confidence |
|-----------------|-----|------------|
| Repetitive outputs | `max_tokens: 2000` | High |
| Thinking exhaustion | `reasoning_effort: low` | High |
| Malformed code blocks | Relaxed regex | High |
| Timeouts | Lower tokens + infrastructure | Medium |

## References

- [OpenRouter Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [LiteLLM Reasoning Content](https://docs.litellm.ai/docs/reasoning_content)
- [LiteLLM Anthropic Effort Parameter](https://docs.litellm.ai/docs/providers/anthropic_effort)
