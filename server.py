"""
KBilliards Web Server — Layer 3 replacement (FastAPI + WebSocket)

Serves the Three.js frontend and runs the physics loop,
communicating ball state to browser clients over WebSocket.
"""

import asyncio
import json
import math
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from controller import BilliardsController
from physics import TABLE_WIDTH, TABLE_LENGTH, BALL_RADIUS
import physics as _phys
from shot_presets import ShotPreset

# ── Controller ──────────────────────────────────────────────────────────────

ctrl = BilliardsController()


# ── Lifespan (startup/shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(game_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

# ── Client / input state ────────────────────────────────────────────────────

clients: list[WebSocket] = []
held_keys: dict[str, bool] = {}

# Scenario map (keys 1-5)
SCENARIOS = {
    "1": (ShotPreset.scenario_1_follow, "1: Follow"),
    "2": (ShotPreset.scenario_2_draw,   "2: Draw"),
    "3": (ShotPreset.scenario_3_stop,   "3: Stop"),
    "4": (ShotPreset.scenario_4_bank,   "4: Bank"),
    "5": (ShotPreset.scenario_5_nejire, "5: Nejire"),
}

# Hit-offset (matches Ursina main.py HitPointUI)
HIT_MAX_OFFSET = 0.025   # meters (~0.76R)
HIT_SPEED = HIT_MAX_OFFSET * 2.0  # WASD speed

# ── Physics params (mirrors PhysicsParamsEditor) ────────────────────────────

PHYSICS_PARAMS = [
    ("MU_SLIDE",            "Slide Frict.",    0.01,   1.0,   0.01),
    ("MU_ROLL",             "Roll Frict.",     0.001,  0.2,   0.002),
    ("MU_SPIN",             "Spin Frict.",     0.001,  0.5,   0.005),
    ("CUSHION_RESTITUTION", "Cushion Rest.",   0.10,   1.0,   0.01),
    ("BALL_RESTITUTION",    "Ball Rest.",      0.10,   1.0,   0.01),
    ("GRAVITY",             "Gravity",         1.0,   20.0,   0.2),
    ("MU_CUSHION",          "Cushion Fric.",   0.0,    0.5,   0.005),
    ("MU_BALL",             "Ball Throw",      0.0,    0.3,   0.005),
    ("SQUIRT_FACTOR",       "Squirt",          0.0,    0.1,   0.005),
    ("RAIL_H_OFFSET",       "Rail Height",     0.0,    0.015, 0.001),
    ("TABLE_BOUNCE_REST",   "Floor Bounce",    0.0,    1.0,   0.01),
]

PARAM_DEFAULTS = {attr: getattr(_phys, attr) for attr, *_ in PHYSICS_PARAMS}

# ── Async game loop ─────────────────────────────────────────────────────────

TARGET_FPS = 60
FRAME_DT = 1.0 / TARGET_FPS


async def game_loop():
    """Main game loop running at ~60 fps."""
    frame_count = 0
    last_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        # Clamp dt to avoid spiral-of-death
        if dt > 0.05:
            dt = 0.05

        # 1. Aiming update (game/practice + idle)
        if (ctrl.game_mode or ctrl.practice_mode) and ctrl.mode == "idle":
            ctrl.update_aim(
                dt,
                rotate_left=held_keys.get("left", False),
                rotate_right=held_keys.get("right", False),
                elev_up=held_keys.get("up", False),
                elev_down=held_keys.get("down", False),
                fine=held_keys.get("shift", False),
            )

        # 2. WASD hit offset
        if (ctrl.game_mode or ctrl.practice_mode) and ctrl.mode == "idle":
            if held_keys.get("w", False):
                ctrl.hit_offset[1] = min(HIT_MAX_OFFSET,
                                         ctrl.hit_offset[1] + HIT_SPEED * dt)
            if held_keys.get("s", False):
                ctrl.hit_offset[1] = max(-HIT_MAX_OFFSET,
                                         ctrl.hit_offset[1] - HIT_SPEED * dt)
            if held_keys.get("a", False):
                ctrl.hit_offset[0] = max(-HIT_MAX_OFFSET,
                                         ctrl.hit_offset[0] - HIT_SPEED * dt)
            if held_keys.get("d", False):
                ctrl.hit_offset[0] = min(HIT_MAX_OFFSET,
                                         ctrl.hit_offset[0] + HIT_SPEED * dt)

        # 3. AI turn
        if ctrl.game_mode and not ctrl.player_turn and ctrl.mode in ("idle", "ai_calculating"):
            ctrl.update_ai_turn(dt)

        # 4. Script tick
        ctrl.tick_script(dt)

        # 5. Physics step
        ctrl.step(dt)

        # 6. Build frame message and broadcast
        if clients:
            frame_msg = _build_frame_message()
            dead: list[WebSocket] = []
            for ws in clients:
                try:
                    await ws.send_text(frame_msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in clients:
                    clients.remove(ws)

        frame_count += 1

        # Sleep to maintain target FPS
        elapsed = time.perf_counter() - now
        sleep_time = FRAME_DT - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            await asyncio.sleep(0)


def _build_frame_message() -> str:
    """Serialize current state into a JSON frame message."""
    # Balls
    balls_data = []
    for b in ctrl.physics_balls:
        balls_data.append({
            "name": b.name,
            "pos": [round(float(b.position[0]), 5),
                    round(float(b.position[1]), 5),
                    round(float(b.position[2]), 5)],
            "w": [round(float(b.angular_velocity[0]), 3),
                  round(float(b.angular_velocity[1]), 3),
                  round(float(b.angular_velocity[2]), 3)],
            "state": b.state.name,
        })

    # Drain pending events
    events = []
    for ev in ctrl.pending_events:
        if ev.get("type") == "spawn_ball" and "ball" in ev:
            b = ev["ball"]
            events.append({
                "type": "spawn_ball",
                "ball": {
                    "name": b.name,
                    "pos": [round(float(b.position[0]), 5),
                            round(float(b.position[1]), 5),
                            round(float(b.position[2]), 5)],
                },
            })
        else:
            events.append(ev)
    ctrl.pending_events.clear()

    # Drain physics events (sounds)
    sounds = []
    for ev in ctrl.physics_events:
        sounds.append({
            "type": ev.get("type", ""),
            "speed": round(float(ev.get("speed", 0.0)), 3),
        })

    # Game state
    game_data = None
    if ctrl.game_mode or ctrl.practice_mode:
        game_data = {
            "active": ctrl.game_mode,
            "practice": ctrl.practice_mode,
            "player_turn": ctrl.player_turn,
            "player_score": ctrl.player_score,
            "ai_score": ctrl.ai_score,
        }

    # Aim state
    aim_data = None
    if ctrl.game_mode or ctrl.practice_mode:
        aim_data = {
            "angle": round(ctrl.aim_angle, 2),
            "elevation": round(ctrl.aim_elevation, 2),
            "power": round(ctrl.aim_power, 4),
            "charging": ctrl.power_charging,
            "hit_offset": [round(ctrl.hit_offset[0], 6),
                           round(ctrl.hit_offset[1], 6)],
        }

    frame = {
        "type": "frame",
        "balls": balls_data,
        "events": events,
        "sounds": sounds,
        "mode": ctrl.mode,
        "status": ctrl.status_msg,
        "info": ctrl.info_msg,
    }
    if game_data is not None:
        frame["game"] = game_data
    if aim_data is not None:
        frame["aim"] = aim_data

    return json.dumps(frame, separators=(',', ':'))


# ── Key press handlers ──────────────────────────────────────────────────────

def _handle_key_down(key: str):
    """Handle a key press event from the client."""
    held_keys[key] = True

    if key == "g":
        ctrl.start_game()
    elif key == "t":
        ctrl.toggle_practice()
    elif key == "r":
        ctrl.reset_game()
    elif key == "space":
        if (ctrl.game_mode or ctrl.practice_mode) and ctrl.mode == "idle":
            ctrl.power_charging = True
    elif key in SCENARIOS:
        fn, label = SCENARIOS[key]
        ctrl.load_scenario(fn, label)


def _handle_key_up(key: str):
    """Handle a key release event from the client."""
    held_keys[key] = False

    if key == "space":
        if ctrl.power_charging:
            ctrl.power_charging = False
            if (ctrl.game_mode or ctrl.practice_mode) and ctrl.mode == "idle":
                ctrl.fire_game_shot()


# ── Physics params helpers ──────────────────────────────────────────────────

def _get_params_data() -> list:
    """Return all physics params with current values."""
    result = []
    for attr, label, mn, mx, step in PHYSICS_PARAMS:
        result.append({
            "attr": attr, "label": label,
            "value": round(getattr(_phys, attr), 6),
            "min": mn, "max": mx, "step": step,
        })
    return result


# ── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)

    # Send init message with table/physics constants
    init_msg = json.dumps({
        "type": "init",
        "table_width": TABLE_WIDTH,
        "table_length": TABLE_LENGTH,
        "ball_radius": BALL_RADIUS,
        "sim_dt": ctrl.SIM_DT,
        "sim_substeps": ctrl.SIM_SUBSTEPS,
        "max_hit_offset": HIT_MAX_OFFSET,
    })
    await ws.send_text(init_msg)

    # Send current ball state as spawn events
    for b in ctrl.physics_balls:
        spawn_msg = json.dumps({
            "type": "frame",
            "balls": [],
            "events": [{
                "type": "spawn_ball",
                "ball": {
                    "name": b.name,
                    "pos": [round(float(b.position[0]), 5),
                            round(float(b.position[1]), 5),
                            round(float(b.position[2]), 5)],
                },
            }],
            "sounds": [],
            "mode": ctrl.mode,
            "status": ctrl.status_msg,
            "info": ctrl.info_msg,
        })
        await ws.send_text(spawn_msg)

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd", "")
            if cmd == "key_down":
                _handle_key_down(msg.get("key", ""))
            elif cmd == "key_up":
                _handle_key_up(msg.get("key", ""))
            elif cmd == "execute":
                ctrl.execute_command(msg.get("text", ""))
            elif cmd == "get_state":
                state_json = ctrl.get_state_json()
                await ws.send_text(json.dumps({
                    "type": "state_json",
                    "data": state_json,
                }))
            elif cmd == "get_params":
                await ws.send_text(json.dumps({
                    "type": "params",
                    "data": _get_params_data(),
                }))
            elif cmd == "adjust_param":
                idx = int(msg.get("index", 0))
                direction = int(msg.get("direction", 0))
                fine = msg.get("fine", False)
                if 0 <= idx < len(PHYSICS_PARAMS):
                    attr, label, mn, mx, step = PHYSICS_PARAMS[idx]
                    s = step / 10.0 if fine else step
                    cur = getattr(_phys, attr)
                    new_val = max(mn, min(mx, cur + direction * s))
                    setattr(_phys, attr, new_val)
                    await ws.send_text(json.dumps({
                        "type": "param_update",
                        "index": idx,
                        "value": round(new_val, 6),
                    }))
            elif cmd == "reset_params":
                for attr, dflt in PARAM_DEFAULTS.items():
                    setattr(_phys, attr, dflt)
                await ws.send_text(json.dumps({
                    "type": "params",
                    "data": _get_params_data(),
                }))
            elif cmd == "set_hit_offset":
                nx = float(msg.get("x", 0.0))
                ny = float(msg.get("y", 0.0))
                ctrl.hit_offset[0] = max(-HIT_MAX_OFFSET,
                                         min(HIT_MAX_OFFSET, nx * HIT_MAX_OFFSET))
                ctrl.hit_offset[1] = max(-HIT_MAX_OFFSET,
                                         min(HIT_MAX_OFFSET, ny * HIT_MAX_OFFSET))
    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)
        held_keys.clear()


# ── Static files + root route ───────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Run with uvicorn ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
