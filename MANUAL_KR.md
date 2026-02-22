# KBilliards — 4-Ball Carom Billiards Simulator

3D 4구 당구(Four-ball Carom) 물리 시뮬레이터.
실시간 물리 엔진, AI 대전, 강화학습(RL) 헤드리스 API를 지원합니다.

---

## 1. 시작하기 (Getting Started)

### 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # macOS/Linux

# 의존성 설치
pip install -r requirements_web.txt
```

### 서버 실행

```bash
python server.py
```

브라우저에서 **http://localhost:8000** 에 접속하면 바로 플레이할 수 있습니다.

### 3-Tier 아키텍처

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

| 계층 | 파일 | 역할 |
|------|------|------|
| **Layer 1** | `physics.py` | 순수 물리 엔진 (마찰, 충돌, 쿠션, 스핀) |
| **Layer 2** | `controller.py` | 게임 로직, 상태 머신, AI, RL API |
| **Layer 3** | `server.py` + `static/` | 웹 서버, 렌더링, 입력 처리 |

Layer 1과 2는 렌더링 프레임워크에 의존하지 않으므로, 헤드리스 RL 학습에 그대로 사용할 수 있습니다.

### 좌표계

```
        +Y (위)
         │
         │
         │
         └──────── +X (오른쪽)
        /
       /
      +Z (전방, 먼 쿠션 방향)
```

- **Y축**: 수직 (위)
- **Z축**: 테이블 길이 방향 (전방 = 먼 쿠션)
- **X축**: 테이블 너비 방향 (오른쪽)

---

## 2. 조작 키 및 단축키 (Controls & Hotkeys)

### 마우스 조작 (OrbitControls)

| 동작 | 조작 |
|------|------|
| 시점 회전 | 좌클릭 드래그 |
| 시점 패닝 | 우클릭 또는 휠클릭 드래그 |
| 줌 인/아웃 | 마우스 휠 스크롤 |
| 당점 조절 | 우하단 원형 UI 드래그 |

### 기본 단축키

| 키 | 기능 |
|----|------|
| `1` ~ `5` | 시나리오 프리셋 로드 (Follow, Draw, Stop, Bank, Nejire) |
| `G` | 게임 모드 시작 (Player vs AI, 10점 선승) |
| `T` | 연습 모드 토글 (자유롭게 샷 연습) |
| `R` | 초기화 (게임 종료 / 연습 리셋 / 공 제거) |
| `P` | 물리 파라미터 에디터 열기/닫기 |
| `Shift` + `P` | 물리 파라미터 전체 기본값 복원 |
| `]` / `[` | 선택된 파라미터 값 증가 / 감소 |
| `Shift` + `A` | 고급 명령어 패널 열기 |

### 샷 조작 키 (게임/연습 모드)

| 키 | 기능 |
|----|------|
| `←` / `→` | 좌우 조준 — Azimuth 회전 (60&deg;/sec) |
| `↑` / `↓` | 큐대 각도 — Elevation 조절 (찍어치기/점프샷용) |
| `W` / `S` | 당점 상하 이동 (탑스핀 / 백스핀) |
| `A` / `D` | 당점 좌우 이동 (좌/우 회전, English) |
| `Shift` | 누른 상태로 조작 시 미세 조정 (1/4 속도) |
| `Space` | **누르고 있으면** 파워 충전, **떼면** 샷 발사 |

#### Azimuth 방향 기준

| 각도 | 방향 |
|------|------|
| 0&deg; | 전방 (+Z, 먼 쿠션) |
| 90&deg; | 오른쪽 (+X) |
| 180&deg; | 후방 (-Z, 가까운 쿠션) |
| 270&deg; | 왼쪽 (-X) |

#### 당점(Tip) 규칙

```
        Top (+Y)
         탑스핀
          │
 Left ────┼──── Right
(-X)      │      (+X)
         백스핀
       Bottom (-Y)
```

- `tip[0]` > 0 = 우측 당점 (Right English) → 좌측 스쿼트 편향
- `tip[0]` < 0 = 좌측 당점 (Left English) → 우측 스쿼트 편향
- `tip[1]` > 0 = 상단 당점 (Top) → 탑스핀 (Follow shot)
- `tip[1]` < 0 = 하단 당점 (Bottom) → 백스핀 (Draw shot)

---

## 3. 고급 명령어 패널 (Advanced Command Panel)

브라우저에서 `Shift + A`를 눌러 패널을 열고, JSON 명령어를 입력합니다.

| 조작 | 동작 |
|------|------|
| `Shift + A` | 패널 열기 (현재 공 상태 자동 로드) |
| `Ctrl + Enter` | 명령어 실행 |
| `Esc` | 패널 닫기 |

### 3.1 공 배치 명령 (set)

임의의 위치와 스핀으로 공을 세팅합니다.

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

| 필드 | 형식 | 설명 |
|------|------|------|
| `pos` | `[x, z]` 또는 `[x, y, z]` | 공 위치 (2D면 y=0 자동) |
| `vel` | `[vx, vz]` 또는 `[vx, vy, vz]` | 초기 속도 (생략 시 정지) |
| `spin` | `[wx, wy, wz]` | 각속도 rad/s (생략 시 0) |
| `state` | `"STATIONARY"` 등 | `STATIONARY`, `SLIDING`, `ROLLING`, `SPINNING` |

### 3.2 물리 파라미터 변경 (set params)

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

### 3.3 정밀 샷 (shot)

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

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `ball` | `"white"` | 타격할 공 이름 |
| `azimuth` | `0` | 수평 조준 각도 (도, 0=전방) |
| `elevation` | `0` | 큐 하향 각도 (도, 찍어치기/점프샷) |
| `tip` | `[0, 0]` | 당점 `[좌우, 상하]`, 범위 -1.0 ~ +1.0 (볼 반지름 비율) |
| `power` | `100` | 타격 세기, MAX_POWER(1.5)의 퍼센트 (0~100) |
| `record` | `false` | `true`이면 궤적을 CSV 파일로 기록 |

### 3.4 상태 저장/불러오기

```json
{"cmd": "save", "file": "my_state"}
{"cmd": "load", "file": "my_state"}
```

파일명은 자동으로 `.json` 확장자가 추가됩니다. `file` 생략 시 타임스탬프 기반 자동 명명.

---

## 4. 물리 파라미터 가이드 (Physics Parameters)

게임 중 `P` 키로 패널을 열어 실시간 튜닝할 수 있습니다.
`]` / `[` 키로 값을 조절하고, `Shift` 를 누르면 1/10 단위 미세 조정됩니다.

### 마찰 계수

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `MU_SLIDE` | 0.20 | 0.01 ~ 1.0 | **슬라이딩 마찰**. 큐 타격 직후 공이 미끄러지다 순수 구름으로 전환되는 속도를 결정. 높을수록 빨리 전환. |
| `MU_ROLL` | 0.015 | 0.001 ~ 0.2 | **구름 마찰**. 순수 구름 상태의 감속률. 높으면 공이 빨리 멈춤. |
| `MU_SPIN` | 0.04 | 0.001 ~ 0.5 | **스핀 감쇠**. 사이드스핀(English, wy)의 자연 감쇠 속도. 높으면 스핀이 빨리 소멸. |

### 반발 계수

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `CUSHION_RESTITUTION` | 0.75 | 0.10 ~ 1.0 | **쿠션 반발**. 1.0이면 완전 탄성(에너지 손실 없음), 낮으면 쿠션에서 큰 에너지 손실. |
| `BALL_RESTITUTION` | 0.94 | 0.10 ~ 1.0 | **공-공 반발**. 두 공 충돌 시 에너지 보존율. 실제 당구공은 0.92~0.96. |
| `TABLE_BOUNCE_REST` | 0.35 | 0.0 ~ 1.0 | **테이블면 반발**. 점프샷 후 공이 테이블에 착지할 때의 반발 계수. |

### 고급 물리 상수

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `MU_CUSHION` | 0.18 | 0.0 ~ 0.5 | **쿠션 마찰 (Coulomb)**. 공이 쿠션에 부딪힐 때 마찰력에 의한 스핀 전달량. 높으면 쿠션 충돌 후 더 많은 회전이 발생. |
| `RAIL_H_OFFSET` | 0.003 | 0.0 ~ 0.015 | **쿠션 접촉 높이** (미터). 공 중심 대비 쿠션이 접촉하는 높이. 이 값이 클수록 쿠션 충돌 시 수직축(wy) 토크가 커져 역회전/탑스핀 효과가 강해짐. |
| `MU_BALL` | 0.08 | 0.0 ~ 0.3 | **공-공 Coulomb 마찰 (Throw)**. 빗맞음 충돌 시 마찰에 의한 타구 분리각 변화. 높으면 "두께 편향(throw)" 효과가 강해짐. |
| `SQUIRT_FACTOR` | 0.02 | 0.0 ~ 0.1 | **스쿼트 편향**. 좌우 당점(English) 사용 시 수구의 초기 진행 방향이 조준선에서 얼마나 벗어나는지. 높으면 사이드 당점의 편향이 커짐. |
| `GRAVITY` | 9.8 | 1.0 ~ 20.0 | **중력 가속도** (m/s&sup2;). 구름 마찰과 슬라이딩 마찰 계산에 사용. |

### 테이블/공 상수 (코드 레벨만 수정 가능)

| 상수 | 값 | 설명 |
|------|----|------|
| `TABLE_WIDTH` | 1.27 m | 테이블 너비 (~50 inch) |
| `TABLE_LENGTH` | 2.54 m | 테이블 길이 (~100 inch) |
| `BALL_RADIUS` | 0.03275 m | 공 반지름 (65.5 mm) |
| `BALL_MASS` | 0.21 kg | 공 질량 |

---

## 5. AI 강화학습용 API 가이드 (Headless RL API)

`controller.py`의 `BilliardsController`는 화면 렌더링 없이 초고속으로 시뮬레이션을 실행할 수 있는 헤드리스 API를 제공합니다.

### 핵심 메서드

#### `reset(balls_state=None) -> np.ndarray`

공을 기본 게임 위치로 초기화하고 관측 벡터를 반환합니다.

```python
from controller import BilliardsController

ctrl = BilliardsController()
obs = ctrl.reset()                                    # 기본 위치
obs = ctrl.reset({"white": {"pos": [0.0, 0.5]}})     # 커스텀 위치
```

기본 게임 위치:

| 공 | 위치 [x, z] |
|----|-------------|
| white (수구) | [-0.25, 0.0] |
| yellow (상대 수구) | [0.25, 0.0] |
| red1 (적구 1) | [0.0, 0.5] |
| red2 (적구 2) | [0.0, -0.5] |

#### `get_obs() -> np.ndarray`

현재 공 상태를 평탄화된 관측 벡터로 반환합니다.

```python
obs = ctrl.get_obs()
# shape: (16,), dtype: float32
# [white_x, white_z, white_vx, white_vz,
#  yellow_x, yellow_z, yellow_vx, yellow_vz,
#  red1_x, red1_z, red1_vx, red1_vz,
#  red2_x, red2_z, red2_vx, red2_vz]
```

#### `simulate_shot(...) -> dict`

**비파괴적** 헤드리스 시뮬레이션. 내부적으로 딥카피를 사용하므로 현재 상태를 변경하지 않습니다.

```python
result = ctrl.simulate_shot(
    ball_name="white",       # 타격할 공
    azimuth=45.0,            # 조준 각도 (도)
    tip=[0.2, 0.5],          # 당점 [좌우, 상하], -1~+1
    power=60.0,              # MAX_POWER의 퍼센트 (0~100)
    elevation=0.0,           # 큐 하향 각도 (도)
    sim_dt=0.005,            # 물리 타임스텝 (기본 0.005)
    max_t=10.0,              # 최대 시뮬레이션 시간 (초)
)
```

**반환값:**

| 키 | 타입 | 설명 |
|----|------|------|
| `scored` | `bool` | 득점 여부 (파울 없이 양쪽 적구 모두 맞힘) |
| `foul` | `bool` | 파울 여부 (상대 수구를 맞힘) |
| `touched` | `list[str]` | 수구가 접촉한 공 이름 목록 |
| `cushion_hits` | `int` | 수구의 쿠션 충돌 횟수 |
| `sim_time` | `float` | 시뮬레이션 경과 시간 (초) |
| `reward` | `float` | 보상 스칼라 (아래 표 참조) |
| `balls` | `dict` | 모든 공의 최종 상태 |
| `obs` | `np.ndarray` | 평탄화된 관측 벡터 (shape 16) |

**보상 체계:**

| 조건 | 보상 |
|------|------|
| 양쪽 적구 모두 맞힘 (득점) | +1.0 |
| 적구 1개만 맞힘 | +0.3 |
| 적구를 하나도 못 맞힘 | -0.1 |
| 파울 (상대 수구 접촉) | -0.5 |

#### `set_balls(balls_info) -> BilliardsController`

시뮬레이션 결과를 현재 상태에 반영합니다 (체이닝 지원).

```python
result = ctrl.simulate_shot("white", azimuth=30, power=70)
ctrl.set_balls(result["balls"])
next_obs = ctrl.get_obs()
```

### OpenAI Gym 연동 예제

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import BilliardsController

class BilliardsEnv(gym.Env):
    """4구 당구 강화학습 환경."""

    def __init__(self):
        super().__init__()
        self.ctrl = BilliardsController()

        # 행동 공간: [azimuth_frac, tip_x, tip_y, power_frac]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.1]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # 관측 공간: 4개 공 × [x, z, vx, vz]
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

        # 결과 반영
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

**학습 루프 예시 (Stable-Baselines3):**

```python
from stable_baselines3 import PPO

env = BilliardsEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

# 학습된 모델 테스트
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(f"reward={reward:.2f}, touched={info['touched']}, scored={done}")
    if done:
        obs, _ = env.reset()
```

### 성능 참고

| 설정 | 속도 |
|------|------|
| `sim_dt=0.005` (기본, RL용) | ~1,000 shots/sec |
| `sim_dt=0.001` (고정밀) | ~200 shots/sec |
| `sim_dt=0.0005` (프리셋 검증용) | ~100 shots/sec |

헤드리스 모드에서는 렌더링 오버헤드가 없으므로, 수십만 에피소드의 대규모 학습이 가능합니다.

---

## 시나리오 프리셋 요약

| 키 | 이름 | 설명 |
|----|------|------|
| `1` | Follow (밀어치기) | 탑스핀으로 수구가 충돌 후 전진 |
| `2` | Draw (끌어치기) | 백스핀으로 수구가 충돌 후 후퇴 |
| `3` | Stop (죽여치기) | 무회전, 수구가 충돌점에서 정지 |
| `4` | Bank (빈쿠션) | 쿠션 반사 후 적구 타격 |
| `5` | Nejire (대회전) | 강타 + English로 3쿠션 이상 경유 |
