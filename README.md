# Four-ball Billiards (Three JS)

3D four-ball carom billiards simulator with real-time physics, AI opponent, and a headless RL API.

Built with **FastAPI** + **Three.js** (WebSocket at ~60 fps).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Realistic physics engine** — sliding/rolling friction, spin (English), cushion rebound, ball-ball throw, masse/jump shots
- **Web-based 3D rendering** — Three.js with orbit camera, power gauge, tip-offset indicator
- **Game mode** — Player vs AI, first to 10 points
- **Practice mode** — free shot practice with full parameter control
- **5 shot presets** — Follow, Draw, Stop, Bank, Nejire (3-cushion)
- **Advanced command panel** — JSON-based ball placement, precision shots, state save/load
- **Physics parameter editor** — tune friction, restitution, and more in real time
- **Headless RL API** — ~1,000 shots/sec for reinforcement learning, OpenAI Gym compatible

---

## Architecture

```
┌──────────────────┐     WebSocket      ┌──────────────┐     ┌──────────────┐
│  Browser Client  │ ◄════════════════► │  FastAPI     │ ──► │  Controller  │
│  (Three.js)      │   JSON frames      │  server.py   │     │  controller  │
│  static/app.js   │   ~60 fps          │  Layer 3     │     │  .py (L2)    │
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
| **Layer 1** | `physics.py` | Pure physics engine (friction, collision, cushion, spin) |
| **Layer 2** | `controller.py` | Game logic, state machine, AI, RL API |
| **Layer 3** | `server.py` + `static/` | Web server, rendering, input handling |

Layers 1 & 2 have no rendering dependency, enabling headless RL training.

---

## Quick Start

```bash
# Clone
git clone https://github.com/skoo1/Four-Ball_Billiards_ThreeJS.git
cd Four-Ball_Billiards_ThreeJS

# Virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # macOS / Linux

# Install dependencies
pip install -r requirements_web.txt

# Run
python server.py
```

Open **http://localhost:8000** in your browser.

---

## Controls

### Mouse

| Action | Input |
|--------|-------|
| Orbit camera | Left-click drag |
| Pan camera | Right-click / wheel-click drag |
| Zoom | Scroll wheel |
| Tip offset | Drag the circle indicator (bottom-right) |

### Keyboard

| Key | Action |
|-----|--------|
| `1`–`5` | Load shot preset (Follow / Draw / Stop / Bank / Nejire) |
| `G` | Start game mode (Player vs AI, first to 10) |
| `T` | Toggle practice mode |
| `R` | Reset |
| `P` | Open/close physics parameter editor |
| `Shift+A` | Open advanced command panel |

### Aiming & Shooting (Game / Practice)

| Key | Action |
|-----|--------|
| `←` / `→` | Aim left / right |
| `↑` / `↓` | Cue elevation (masse / jump) |
| `W` / `S` | Tip offset up / down (topspin / backspin) |
| `A` / `D` | Tip offset left / right (English) |
| `Shift` | Fine adjustment (1/4 speed) |
| `Space` | Hold to charge power, release to shoot |

---

## Headless RL API

The controller provides a rendering-free API for reinforcement learning.

```python
from controller import BilliardsController

ctrl = BilliardsController()
obs = ctrl.reset()  # shape (16,): [x, z, vx, vz] × 4 balls

result = ctrl.simulate_shot(
    ball_name="white",
    azimuth=45.0,       # degrees, 0 = forward (+Z)
    tip=[0.2, 0.5],     # [side, vertical], -1 to +1
    power=60.0,          # 0–100 (% of max)
)

print(result["scored"], result["reward"])
ctrl.set_balls(result["balls"])  # apply result
```

**Performance:** ~1,000 shots/sec (`sim_dt=0.005`).

See [MANUAL_KR.md](MANUAL_KR.md) or [MANUAL_ENG.md](MANUAL_ENG.md) for the full API reference and an OpenAI Gym integration example.

---

## Project Structure

```
Four-Ball_Billiards_ThreeJS/
├── physics.py            # Layer 1 — physics engine
├── controller.py         # Layer 2 — game logic, AI, RL API
├── server.py             # Layer 3 — FastAPI WebSocket server
├── shot_presets.py       # 5 built-in shot scenarios
├── main.py               # Ursina-based desktop client (standalone)
├── static/
│   ├── index.html        # Web UI
│   └── app.js            # Three.js renderer & input
├── scripts/              # Example shot scripts
├── tests/                # pytest test suite
├── docs/                 # Lecture notes & physics documentation
├── requirements_web.txt  # Python dependencies (FastAPI, uvicorn, numpy)
├── MANUAL_KR.md          # Full manual (Korean)
└── MANUAL_ENG.md         # Full manual (English)
```

---

## Dependencies

- Python 3.10+
- FastAPI + Uvicorn (web server)
- NumPy (physics computations)
- Three.js (loaded via CDN, no npm required)

---

## Documentation

- [MANUAL_KR.md](MANUAL_KR.md) — Full user manual (Korean)
- [MANUAL_ENG.md](MANUAL_ENG.md) — Full user manual (English)
- [docs/](docs/) — Physics lecture notes and derivations

---

## Author & Acknowledgments

Made by Seungbum Koo, PhD KAIST, Daejeon, South Korea

Developed with the assistance of Claude Code.

---

## License

MIT