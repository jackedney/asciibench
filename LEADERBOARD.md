# ASCIIBench Leaderboard

Model rankings based on Elo ratings from pairwise comparisons.

## Rankings

| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |
|------|-------|------------|-------------|----------|--------------|
| - | - | - | - | - | - |

## Methodology

This leaderboard uses the Elo rating system to rank models based on
pairwise vote data from the Judge UI. Higher ratings indicate better
performance in ASCII art generation tasks.

### Elo Rating Details

- **Base Rating**: 1500
- **K-Factor**: 32
- **Minimum Comparisons**: 10 (to appear on leaderboard)

For more details on the Elo calculation approach, see
`asciibench/analyst/elo.py`.
