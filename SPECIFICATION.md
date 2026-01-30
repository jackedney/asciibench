# Specification: ASCIIBench (Alpha)

## 1. Overview
ASCIIBench is a human-in-the-loop, Elo-based benchmark designed to rank Large Language Models (LLMs) based on their spatial reasoning and artistic capability within the medium of ASCII art.

## 2. Core Architecture
The system consists of three distinct modules:
1.  **The Generator:** A batch processor that uses OpenRouter to generate samples.
2.  **The Judge UI:** A local double-blind web interface for 1v1 comparisons.
3.  **The Analyst:** A script to calculate Elo ratings and generate the final leaderboard.

## 3. Data Definitions

### 3.1 `models.yaml`
A list of OpenRouter model identifiers.
```yaml
models:
  - openai/gpt-4o
  - anthropic/claude-3.5-sonnet
  - meta-llama/llama-3.1-70b-instruct
  # ... etc
```

### 3.2 `prompts.yaml` (Templates)
The system will generate 10 unique prompts for each of the 4 categories using templates.
*   **Category 1: Single Object** (`"Draw a [OBJECT] in ASCII art"`)
*   **Category 2: Single Animal** (`"Draw a [ANIMAL] in ASCII art"`)
*   **Category 3: Animal + Action** (`"Draw a [ANIMAL] [ACTION] in ASCII art"`)
*   **Category 4: Spatial Relationship** (`"Draw a [OBJECT_A] [POSITION] a [OBJECT_B] in ASCII art"`)

### 3.3 `config.yaml`
```yaml
generation:
  attempts_per_prompt: 5
  temperature: 0.0
  max_tokens: 1000
  provider: openrouter
  system_prompt: |
    Draw the requested ASCII art. 
    Wrap your art in a Markdown code block.
    Output ONLY the art. No conversational text.
    Do not use color or ANSI escape codes.
```

## 4. Execution Pipeline

### Phase 1: Generation (Batch)
1.  The Generator expands templates into 40 unique prompts.
2.  For every model in `models.yaml`, it calls the API 5 times per prompt.
3.  **Output:** A `database.jsonl` where each entry contains:
    *   `model_id`, `prompt_text`, `category`, `attempt_number`, `raw_output`, `sanitized_output` (extracted from markdown).
4.  **Safety Valve:** If an output exceeds 1000 tokens or lacks a markdown block, it is flagged as `invalid`.

### Phase 2: Evaluation (Double-Blind)
The web UI presents a random 1v1 matchup:
1.  **Double Blind:** Model names are hidden.
2.  **Triple-Font Rendering:** To ensure visual robustness, each candidate is rendered three times in a vertical stack using different font families:
    *   **Font 1 (Classic):** `Courier New, monospace`
    *   **Font 2 (Modern):** `Consolas, Monaco, monospace`
    *   **Font 3 (Condensed/Stylized):** `Fira Code, Lucida Console, monospace`
3.  **Judge Input (Keyboard):**
    *   `[A]` Left is better.
    *   `[D]` Right is better.
    *   `[S]` Tie (Both are good).
    *   `[F]` Fail (Both are terrible).
4.  **Persistence:** Judgments are saved to `votes.jsonl`.

### Phase 3: Ranking
1.  The Analyst reads `votes.jsonl`.
2.  It applies the **Bradley-Terry model** (or standard Elo) to calculate scores.
3.  It generates `LEADERBOARD.md` with categories:
    *   **Overall Elo**
    *   **Category-Specific Elo** (Which model is the "Animal King"?)
    *   **Consistency Score** (Variance across the 5 attempts).

## 5. Success Metrics
*   **Primary:** Stable Elo ranking with a 95% confidence interval.
*   **Secondary:** Visual robustness (art that looks good in all 3 fonts vs. art that "breaks" in one).
