# CLI Loading Animation Redesign

## Overview

Redesign the CLI loading display to be more vibrant with a cleaner layout, smooth gradient progress bar, and a domino run ambient animation.

## Visual Layout

```
✓ 3 | ✗ 1 | $0.06      ← running totals (updates in-place)
  claude-3-opus         ← model name with runescape color cycling
  ━━━━━━━━━━━─────────  ← smooth rainbow gradient progress bar
  ▌▞▄▄▄▄▄▄▄▄            ← domino run animation
```

## Components

### 1. Running Totals Status Line

- Single line at the top that updates in-place
- Shows accumulated counts: `✓ {success} | ✗ {failed} | ${cost}`
- Replaces current behavior of stacking new status lines after each model

### 2. Model Name Display

- Single instance of model name (no repeating pattern)
- Runescape-style discrete color cycling through rainbow spectrum
- Colors jump between characters, shifting each frame
- Centered above the progress bar

### 3. Smooth Rainbow Gradient Progress Bar

- Minimalist thin line using `━` (filled) and `─` (empty)
- Smooth rainbow gradient across the filled portion
- Colors blend between rainbow spectrum (not discrete jumps like the model name)
- RGB interpolation: for position `p` between color A and B, blend as `A * (1-p) + B * p`
- Width matches current loader width

### 4. Domino Run Animation

**Characters:**
- `▌` - standing
- `▞` - tipping right / rising from right fall
- `▚` - tipping left / rising from left fall
- `▄` - fallen (flat)

**The Fall Sequence (left-to-right):**
```
▌ → ▞ → ▄
```

**The Fall Sequence (right-to-left):**
```
▌ → ▚ → ▄
```

**The Rise Sequence:**
All dominoes rise together through 3 frames:
```
▄ → ▞/▚ → ▌
```

**Movement Pattern:**

1. Left-to-right fall: Dominoes tip over one-by-one from left to right, each staying fallen
2. All rise together: Once all are flat, they all rise back to standing simultaneously (3 frames)
3. Right-to-left fall: Dominoes tip over one-by-one from right to left
4. All rise together: Back to standing simultaneously (3 frames)
5. Repeat from step 1

**Visual Example (full cycle):**
```
Left-to-right fall:
  ▞▌▌▌▌▌▌▌▌▌
  ▄▞▌▌▌▌▌▌▌▌
  ▄▄▞▌▌▌▌▌▌▌
  ...
  ▄▄▄▄▄▄▄▄▄▄  (all fallen)

All rise together:
  ▞▞▞▞▞▞▞▞▞▞  (all rising)
  ▌▌▌▌▌▌▌▌▌▌  (all standing)

Right-to-left fall:
  ▌▌▌▌▌▌▌▌▌▚
  ▌▌▌▌▌▌▌▌▚▄
  ...
  ▄▄▄▄▄▄▄▄▄▄  (all fallen)

All rise together:
  ▚▚▚▚▚▚▚▚▚▚  (all rising)
  ▌▌▌▌▌▌▌▌▌▌  (all standing)

Repeat...
```

**Timing:**
- Matches full width of progress bar
- Synchronized with same frame rate as color cycling (12 FPS default)

## Implementation

### File Changes

| File | Change |
|------|--------|
| `asciibench/common/wave_text.py` | Add smooth gradient function for progress bar |
| `asciibench/common/loader.py` | Refactor `RuneScapeLoader` to render new 4-layer layout |
| `asciibench/common/domino.py` (new) | Domino animation state machine |

### Gradient Implementation

```python
def interpolate_color(color_a: tuple, color_b: tuple, t: float) -> tuple:
    """Blend between two RGB colors. t=0 gives color_a, t=1 gives color_b."""
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(color_a, color_b))
```

Rainbow colors for gradient:
- Red (255, 0, 0)
- Orange (255, 165, 0)
- Yellow (255, 255, 0)
- Green (0, 255, 0)
- Cyan (0, 255, 255)
- Blue (0, 0, 255)
- Purple (128, 0, 128)

### Domino State Machine

```python
@dataclass
class DominoState:
    direction: Literal["left_to_right", "right_to_left"]
    phase: Literal["falling", "rising"]
    position: int  # Current falling domino index, or rise frame (0-2)
    width: int
```

### Status Line Update

Modify the loader to maintain running totals instead of printing new lines:
- Track `success_count`, `failure_count`, `total_cost`
- Update single Rich Text element in-place
- Increment counts on each model completion

### Frame Sync

All four elements update on the same frame tick:
- Status line (when values change)
- Model name color cycling
- Progress bar (when progress changes)
- Domino animation

Maintains current 12 FPS default.
