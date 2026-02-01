# Loader and Animation Code Location

## Story US-001: Locate loader and animation code

This document documents the location of all loader UI code as identified during US-001.

### Files Identified

#### 1. Main Loader Component
**File:** `asciibench/common/loader.py` (535 lines)

Primary loader implementation using Rich Live for non-blocking animation updates.

Key components:
- `RuneScapeLoader` class (line 133): Main loader class with 4-layer layout
- `_render_frame()` (line 384): Renders the current animation frame with 4 layers:
  1. Status line
  2. Model name (centered)
  3. Progress bar with gradient
  4. Domino row (ambient animation)
- `update()` (line 234): Updates progress to given step
- `complete()` (line 293): Shows green/red flash effect on completion
- `start()` (line 465): Starts the animated loader
- `stop()` (line 496): Stops the animated loader

#### 2. Status Line Rendering
**File:** `asciibench/common/wave_text.py` (298 lines)

Status line implementation showing success/failure counts and total cost.

Key function:
- `render_status_line()` (line 260): Renders status line with format '✓ {success} | ✗ {failed} | ${cost:.4f}'
  - Success styled green
  - Failure styled red

Used in: `loader.py` line 410

#### 3. Domino Animation
**File:** `asciibench/common/domino.py` (185 lines)

Ambient falling/rising domino effect animation.

Key components:
- `DominoState` dataclass (line 14): State tracking for domino animation
  - direction: "left_to_right", "right_to_left", or "rising"
  - phase: Current animation phase (0-2)
  - position: Current position for falling animations
  - width: Number of dominoes in animation
- `get_domino_frame()` (line 97): Gets next frame of domino animation
- `_render_left_to_right()` (line 30): Renders left-to-right falling animation
- `_render_right_to_left()` (line 55): Renders right-to-left falling animation
- `_render_rising()` (line 80): Renders rising animation

Domino characters used:
- `▌` (U+258C): Standing
- `▞` (U+259E): Tipping right
- `▚` (U+259A): Tipping left
- `▄` (U+2584): Fallen

Used in: `loader.py` lines 189-191 (initialization), 429-434 (rendering)

#### 4. Gradient/Loading Bar
**File:** `asciibench/common/wave_text.py` (298 lines)

Rainbow gradient rendering for the loading bar.

Key components:
- `render_gradient_text()` (line 163): Renders text with rainbow gradient
  - Applies gradient only to filled portion based on filled_ratio
  - Unfilled portion uses dim style
  - Gradient flows and shifts based on frame number
- `interpolate_color()` (line 124): Interpolates between two hex colors
- `RAINBOW_COLORS` constant (line 8): Color sequence
  - Red (#FF0000)
  - Orange (#FF7F00)
  - Yellow (#FFFF00)
  - Green (#00FF00)
  - Cyan (#00FFFF)
  - Blue (#0000FF)
  - Purple (#8B00FF)

Used in: `loader.py` line 426

#### 5. Model Name Display
**File:** `asciibench/common/loader.py` (535 lines)

Model name rendering with color cycling effect.

Location: `_render_frame()` method, lines 412-418

Current implementation:
```python
model_name_text = render_cycling_text(model_name, frame, shift_interval=2)
model_name_len = len(model_name)
left_pad = (width - model_name_len) // 2
if left_pad > 0:
    model_name_text = Text(" " * left_pad) + model_name_text
```

Note: Model name is currently CENTERED using left_pad calculation.

Related function in `wave_text.py`:
- `render_cycling_text()` (line 219): Renders text with discrete color cycling
  - Each character gets a color from rainbow sequence
  - Color pattern shifts every shift_interval frames (default 2)

#### 6. Main Entry Point (Usage Context)
**File:** `asciibench/generator/main.py` (278 lines)

Shows how the loader is used in the generator workflow.

Key usage:
- Line 22: Imports RuneScapeLoader
- Line 24: Imports create_loader from simple_display
- Lines 157-195: `_loader_progress_callback()` - Progress callback using RuneScape loader
  - Detects model changes
  - Shows stats line (SEPARATE from loader's status line) using `show_stats()` at line 185
  - Creates new loader for each model batch at line 189
- Lines 221-223: Completes and stops final loader

**Note:** There is a potential duplicate status line issue:
- The loader has its own status line (rendered in `_render_frame()` line 410)
- But `main.py` also shows a stats line using `show_stats()` at line 185, which prints SEPARATELY

#### 7. Simplified Display Wrapper
**File:** `asciibench/common/simple_display.py` (72 lines)

Factory wrapper for creating loader instances.

Key function:
- `create_loader()` (line 51): Factory function to create RuneScapeLoader
  - Passes shared console instance for consistent display

### Summary Table

| Component | File | Line(s) | Notes |
|-----------|------|---------|-------|
| Status line rendering | wave_text.py | 260 | Function: render_status_line() |
| Dominoes animation | domino.py | 14-184 | DominoState class + render functions |
| Gradient/loading bar | wave_text.py | 163, 124 | render_gradient_text(), interpolate_color() |
| Model name display | loader.py | 412-418 | Currently CENTERED (left_pad calculation) |
| Main loader class | loader.py | 133-535 | RuneScapeLoader class with 4-layer layout |
| Loader usage | generator/main.py | 157-223 | Progress callback and stats display |

### Search Terms Used

The following search terms were used to locate the loader code:
- "status.*line"
- "progress.*bar"
- "loading"
- "loader"
- "domino"
- "animation"
- "gradient"

All search terms successfully identified relevant code.
