```
 █████╗ ███████╗ ██████╗██╗██╗██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██║██║██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
███████║███████╗██║     ██║██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██║╚════██║██║     ██║██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██║  ██║███████║╚██████╗██║██║██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝╚═╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
```

**Elo-based benchmark for ranking LLMs on spatial reasoning through ASCII art.**

---

## Quick Start

```bash
uv sync
cp .env.example .env  # add OPENROUTER_API_KEY
uv run dev            # → http://localhost:8000
```

---

## How It Works

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Generator  │ ───▶ │   Judge UI   │ ───▶ │   Analyst    │
│              │      │              │      │              │
│  batch LLM   │      │  double-blind│      │  Elo ratings │
│  generation  │      │  comparison  │      │  leaderboard │
└──────────────┘      └──────────────┘      └──────────────┘
        ↓                    ↓                      ↓
   database.jsonl       votes.jsonl          LEADERBOARD.md
```

---

## Commands

| Command | Description |
|---------|-------------|
| `uv run dev` | Start judge UI at localhost:8000 |
| `uv run generate` | Generate ASCII samples from configured models |
| `uv run analyze` | Calculate Elo ratings and generate leaderboard |

---

## Configuration

| File | Purpose |
|------|---------|
| `models.yaml` | OpenRouter model identifiers |
| `prompts.yaml` | Prompt templates by category |
| `config.yaml` | Generation settings (temperature, tokens, etc.) |

---

## Judge Interface

Keyboard-driven for speed:

| Key | Action |
|-----|--------|
| `A` | Left wins |
| `D` | Right wins |
| `S` | Tie |
| `F` | Both fail |
| `Z` | Undo |

Samples rendered in 3 fonts to test robustness across typography.

---

<details>
<summary><strong>Development</strong></summary>

```bash
uv run pytest --cov    # tests
uv run ruff check      # lint
uv run ruff format     # format
uv run ty check        # types
```

</details>

<details>
<summary><strong>Observability (Logfire)</strong></summary>

Optional. Add to `.env`:
```bash
LOGFIRE_TOKEN=your_token
```

Captures request/response traces, token usage, and costs.

</details>

---

[Specification](SPECIFICATION.md)
