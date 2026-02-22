# KBilliards — 4-Ball Carom Billiards Simulator

A 3D four-ball carom billiards physics simulator.
Features a real-time physics engine, AI opponent, and a headless reinforcement learning (RL) API.

---

## 1. Getting Started

### Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements_web.txt
```

### Running the Server

```bash
python server.py
```

Open **http://localhost:8000** in your browser to start playing.

### 3-Tier Architecture

```
┌──────────────────┐     WebSocket      ┌──────────────┐     ┌──────────────┐
│  Browser Client  │ ◄════════════════► │  FastAPI     │ ──► │  Controller  │
│  (Three.js)      │   JSON frames      │  server.py   │     │  controller  │
│  static/app.js   │   ~60 fps          │  Layer 3     │     │  .py (L2)    │
│  static/index.   │                    │              │     │              │
│  html            │                    │              │     │              │
└──────────────────┘                    └──────────────┘     └──────┬───────┘
                                                                    │
                                                              ┌─────▼───────┐
                                                              │  Physics    │
                                                              │  Engine     │
                                                              │  physics.py │
                                                              │  (L1)       │
                                                              └─────────────┘
```

| Layer | File | Role |
|-------|------|------|
| **Layer 1** | `physics.py` | Pure physics engine (friction, collisions, cushions, spin) |
| **Layer 2** | `controller.py` | Game logic, state machine, AI, RL API |
| **Layer 3** | `server.py` + `static/` | Web server, rendering, input handling |

Layers 1 and 2 have no rendering dependencies, making them directly usable for headless RL training.

### Coordinate System

```
        +Y (Up)
         │
         │
         │
         └──────── +X (Right)
        /
       /
      +Z (Forward, toward far cushion)
```

- **Y-axis**: Vertical (up)
- **Z-axis**: Table length (forward = far cushion)
- **X-axis**: Table width (right)

---

## 2. Controls & Hotkeys

### Mouse Controls (OrbitControls)

| Action | Input |
|--------|-------|
| Rotate view | Left-click drag |
| Pan view | Right-click or middle-click drag |
| Zoom in/out | Mouse wheel scroll |
| Adjust hit point | Drag the circle UI (bottom-right) |

### General Hotkeys

| Key | Function |
|-----|----------|
| `1` ~ `5` | Load scenario preset (Follow, Draw, Stop, Bank, Nejire) |
| `G` | Start game mode (Player vs AI, first to 10 points) |
| `T` | Toggle practice mode (free shot practice) |
| `R` | Reset (quit game / restart practice / clear balls) |
| `P` | Toggle physics parameter editor |
| `Shift` + `P` | Reset all physics parameters to defaults |
| `]` / `[` | Increase / decrease selected parameter value |
| `Shift` + `A` | Open Advanced Command Panel |

### Shot Controls (Game / Practice Mode)

| Key | Function |
|-----|----------|
| `←` / `→` | Aim left/right — Azimuth rotation (60&deg;/sec) |
| `↑` / `↓` | Cue angle — Elevation control (for masse/jump shots) |
| `W` / `S` | Move hit point up/down (topspin / backspin) |
| `A` / `D` | Move hit point left/right (left/right English) |
| `Shift` | Hold for fine control (1/4 speed) |
| `Space` | **Hold** to charge power, **release** to fire |

#### Azimuth Reference

| Angle | Direction |
|-------|-----------|
| 0&deg; | Forward (+Z, far cushion) |
| 90&deg; | Right (+X) |
| 180&deg; | Backward (-Z, near cushion) |
| 270&deg; | Left (-X) |

#### Hit Point (Tip) Convention

```
        Top (+Y)
        Topspin
          │
 Left ────┼──── Right
(-X)      │      (+X)
        Backspin
       Bottom (-Y)
```

- `tip[0]` > 0 = Right English → squirt deflection to the left
- `tip[0]` < 0 = Left English → squirt deflection to the right
- `tip[1]` > 0 = Top hit → topspin (follow shot)
- `tip[1]` < 0 = Bottom hit → backspin (draw shot)

---

## 3. Advanced Command Panel

Press `Shift + A` in the browser to open the panel and enter JSON commands.

| Input | Action |
|-------|--------|
| `Shift + A` | Open panel (current ball state auto-loaded) |
| `Ctrl + Enter` | Execute command |
| `Esc` | Close panel |

### 3.1 Ball Placement (set)

Place balls at arbitrary positions with optional velocity and spin.

```json
{
  "cmd": "set",
  "balls": {
    "white":  { "pos": [0.0, -0.6] },
    "yellow": { "pos": [0.25, 0.0] },
    "red1":   { "pos": [0.0, 0.5], "vel": [0.5, 0.0] },
    "red2":   { "pos": [-0.3, 0.8], "spin": [0.0, 30.0, 0.0] }
  }
}
```

| Field | Format | Description |
|-------|--------|-------------|
| `pos` | `[x, z]` or `[x, y, z]` | Ball position (y defaults to 0 for 2D) |
| `vel` | `[vx, vz]` or `[vx, vy, vz]` | Initial velocity (defaults to stationary) |
| `spin` | `[wx, wy, wz]` | Angular velocity in rad/s (defaults to 0) |
| `state` | `"STATIONARY"` etc. | `STATIONARY`, `SLIDING`, `ROLLING`, `SPINNING` |

### 3.2 Physics Parameter Override (set params)

```json
{
  "cmd": "set",
  "params": {
    "MU_ROLL": 0.015,
    "MU_CUSHION": 0.18,
    "GRAVITY": 9.81
  }
}
```

### 3.3 Precision Shot (shot)

```json
{
  "cmd": "shot",
  "ball": "white",
  "azimuth": 30.0,
  "elevation": 0.0,
  "tip": [-0.2, 0.5],
  "power": 80,
  "record": true
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `ball` | `"white"` | Ball to strike |
| `azimuth` | `0` | Horizontal aim angle (degrees, 0 = forward) |
| `elevation` | `0` | Downward cue angle (degrees, for masse/jump shots) |
| `tip` | `[0, 0]` | Hit point `[left-right, top-bottom]`, range -1.0 to +1.0 (ball radius fraction) |
| `power` | `100` | Shot force as percentage of MAX_POWER (1.5), range 0-100 |
| `record` | `false` | If `true`, records ball trajectories to a CSV file |

### 3.4 Save / Load State

```json
{"cmd": "save", "file": "my_state"}
{"cmd": "load", "file": "my_state"}
```

The `.json` extension is added automatically. If `file` is omitted, a timestamp-based name is generated.

---

## 4. Physics Parameters Guide

Press `P` during gameplay to open the parameter editor.
Use `]` / `[` to adjust values. Hold `Shift` for fine adjustment (1/10 step).

### Friction Coefficients

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `MU_SLIDE` | 0.20 | 0.01 ~ 1.0 | **Sliding friction.** Controls how quickly a ball transitions from sliding to pure rolling after being struck. Higher values cause faster transition. |
| `MU_ROLL` | 0.015 | 0.001 ~ 0.2 | **Rolling friction.** Deceleration rate during pure rolling. Higher values cause balls to stop sooner. |
| `MU_SPIN` | 0.04 | 0.001 ~ 0.5 | **Spin decay.** Natural decay rate of sidespin (English, wy). Higher values cause spin to dissipate faster. |

### Restitution Coefficients

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `CUSHION_RESTITUTION` | 0.75 | 0.10 ~ 1.0 | **Cushion bounce.** 1.0 = perfectly elastic (no energy loss); lower values cause greater energy loss at cushions. |
| `BALL_RESTITUTION` | 0.94 | 0.10 ~ 1.0 | **Ball-ball bounce.** Energy retention in ball-ball collisions. Real billiard balls are typically 0.92-0.96. |
| `TABLE_BOUNCE_REST` | 0.35 | 0.0 ~ 1.0 | **Table surface bounce.** Restitution when a ball lands on the table after a jump shot. |

### Advanced Physics Constants

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `MU_CUSHION` | 0.18 | 0.0 ~ 0.5 | **Cushion friction (Coulomb).** Controls spin transfer when a ball strikes a cushion. Higher values produce more post-cushion spin. |
| `RAIL_H_OFFSET` | 0.003 | 0.0 ~ 0.015 | **Cushion contact height** (meters). Height above ball center where the cushion makes contact. Larger values increase vertical-axis (wy) torque during cushion hits, strengthening reverse spin/topspin effects. |
| `MU_BALL` | 0.08 | 0.0 ~ 0.3 | **Ball-ball Coulomb friction (Throw).** Controls departure angle deviation in off-center collisions. Higher values produce stronger "throw" effects. |
| `SQUIRT_FACTOR` | 0.02 | 0.0 ~ 0.1 | **Squirt deflection.** Controls how much the cue ball's initial trajectory deviates from the aim line when using left/right English. Higher values increase sidespin deflection. |
| `GRAVITY` | 9.8 | 1.0 ~ 20.0 | **Gravitational acceleration** (m/s&sup2;). Used in rolling and sliding friction calculations. |

### Table & Ball Constants (Code-Level Only)

| Constant | Value | Description |
|----------|-------|-------------|
| `TABLE_WIDTH` | 1.27 m | Table playing surface width (~50 inches) |
| `TABLE_LENGTH` | 2.54 m | Table playing surface length (~100 inches) |
| `BALL_RADIUS` | 0.03275 m | Ball radius (65.5 mm) |
| `BALL_MASS` | 0.21 kg | Ball mass |

---

## 5. Headless RL API Guide

The `BilliardsController` in `controller.py` provides a headless API for running ultra-fast simulations without any rendering.

### Core Methods

#### `reset(balls_state=None) -> np.ndarray`

Resets balls to default game positions and returns the observation vector.

```python
from controller import BilliardsController

ctrl = BilliardsController()
obs = ctrl.reset()                                    # Default positions
obs = ctrl.reset({"white": {"pos": [0.0, 0.5]}})     # Custom position
```

Default game positions:

| Ball | Position [x, z] |
|------|-----------------|
| white (cue ball) | [-0.25, 0.0] |
| yellow (opponent cue ball) | [0.25, 0.0] |
| red1 (target 1) | [0.0, 0.5] |
| red2 (target 2) | [0.0, -0.5] |

#### `get_obs() -> np.ndarray`

Returns the current ball state as a flattened observation vector.

```python
obs = ctrl.get_obs()
# shape: (16,), dtype: float32
# [white_x, white_z, white_vx, white_vz,
#  yellow_x, yellow_z, yellow_vx, yellow_vz,
#  red1_x, red1_z, red1_vx, red1_vz,
#  red2_x, red2_z, red2_vx, red2_vz]
```

#### `simulate_shot(...) -> dict`

**Non-destructive** headless simulation. Uses deep copies internally, leaving the current state unchanged.

```python
result = ctrl.simulate_shot(
    ball_name="white",       # Ball to strike
    azimuth=45.0,            # Aim angle (degrees)
    tip=[0.2, 0.5],          # Hit point [left-right, top-bottom], -1 to +1
    power=60.0,              # Percentage of MAX_POWER (0-100)
    elevation=0.0,           # Downward cue angle (degrees)
    sim_dt=0.005,            # Physics timestep (default 0.005)
    max_t=10.0,              # Maximum simulation time (seconds)
)
```

**Return value:**

| Key | Type | Description |
|-----|------|-------------|
| `scored` | `bool` | Whether a point was scored (no foul + both target balls hit) |
| `foul` | `bool` | Whether a foul occurred (opponent's cue ball was hit) |
| `touched` | `list[str]` | Names of all balls contacted by the cue ball |
| `cushion_hits` | `int` | Number of cushion hits by the cue ball |
| `sim_time` | `float` | Elapsed simulation time (seconds) |
| `reward` | `float` | Scalar reward (see table below) |
| `balls` | `dict` | Final state of all balls |
| `obs` | `np.ndarray` | Flattened observation vector (shape 16) |

**Reward structure:**

| Condition | Reward |
|-----------|--------|
| Both target balls hit (score) | +1.0 |
| Only one target ball hit | +0.3 |
| No target ball hit (miss) | -0.1 |
| Foul (opponent's cue ball contacted) | -0.5 |

#### `set_balls(balls_info) -> BilliardsController`

Applies simulation results to the current state (supports method chaining).

```python
result = ctrl.simulate_shot("white", azimuth=30, power=70)
ctrl.set_balls(result["balls"])
next_obs = ctrl.get_obs()
```

### OpenAI Gym Integration Example

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import BilliardsController

class BilliardsEnv(gym.Env):
    """Four-ball carom billiards RL environment."""

    def __init__(self):
        super().__init__()
        self.ctrl = BilliardsController()

        # Action space: [azimuth_frac, tip_x, tip_y, power_frac]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.1]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Observation space: 4 balls x [x, z, vx, vz]
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(16,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.ctrl.reset()
        return obs, {}

    def step(self, action):
        azimuth = action[0] * 360.0
        tip = [float(action[1]), float(action[2])]
        power = float(action[3]) * 100.0

        result = self.ctrl.simulate_shot(
            "white", azimuth=azimuth, tip=tip, power=power,
        )

        # Apply result
        self.ctrl.set_balls(result["balls"])
        obs = self.ctrl.get_obs()
        reward = result["reward"]
        terminated = result["scored"]

        return obs, reward, terminated, False, {
            "touched": result["touched"],
            "cushion_hits": result["cushion_hits"],
            "foul": result["foul"],
        }
```

**Training loop example (Stable-Baselines3):**

```python
from stable_baselines3 import PPO

env = BilliardsEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

# Test the trained model
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(f"reward={reward:.2f}, touched={info['touched']}, scored={done}")
    if done:
        obs, _ = env.reset()
```

### Performance Notes

| Setting | Throughput |
|---------|-----------|
| `sim_dt=0.005` (default, for RL) | ~1,000 shots/sec |
| `sim_dt=0.001` (high precision) | ~200 shots/sec |
| `sim_dt=0.0005` (preset validation) | ~100 shots/sec |

In headless mode there is no rendering overhead, enabling large-scale training over hundreds of thousands of episodes.

---

## Scenario Presets Summary

| Key | Name | Description |
|-----|------|-------------|
| `1` | Follow | Topspin causes cue ball to continue forward after collision |
| `2` | Draw | Backspin causes cue ball to return backward after collision |
| `3` | Stop | No spin — cue ball stops at the collision point |
| `4` | Bank | Cushion rebound followed by target ball contact |
| `5` | Nejire | Power shot + English for 3+ cushion traversal |
