---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 1. 핵심 물리 엔진 구현
**1교시: 강체 역학 기초 및 공의 상태 정의 (Stationary, Rolling, Sliding)**

---

## 🎯 이번 시간의 목표

당구공은 단순한 '점 질량(Point Mass)'이 아닙니다. 회전(Spin)을 가지는 **강체(Rigid Body)**입니다.

1. **강체 역학의 기초:** 질량 중심의 이동(선속도)과 회전(각속도)의 분리
2. **유한 상태 기계(FSM):** 당구공이 가지는 3가지 물리적 상태 이해
3. **마찰력(Coulomb Friction):** 공은 왜 미끄러지다가 구르기 시작하는가?

---

## 1. 당구공의 물리적 상태 (State Machine)

실제 당구공은 큐에 맞는 순간부터 멈출 때까지 **3가지 상태**를 전이합니다.
우리 물리 엔진(`physics.py`)에서는 이를 `Enum`으로 명확히 정의합니다.

```python
import enum

class BallState(enum.Enum):
    STATIONARY = 0  # 정지 상태
    ROLLING = 1     # 구름 상태 (미끄러짐 없이 바닥을 움켜쥐고 구름)
    SLIDING = 2     # 미끄러짐 상태 (선속도와 각속도가 불일치하여 바닥을 스치며 이동)
```

> **👨‍🏫 Instructor Note:**
> 처음 큐로 공을 강하게 치면 공은 회전 없이 바닥을 주욱 미끄러지며 나아갑니다(`SLIDING`). 그러다 바닥 마찰에 의해 점차 회전이 발생하고, 미끄러짐이 멈추면 완벽히 굴러가게 됩니다(`ROLLING`).

---

## 2. 공 객체(Ball)의 데이터 구조 설계

3D 공간에서 강체를 표현하려면 **위치(Position)**와 **선속도(Velocity)** 뿐만 아니라, **회전 각속도(Angular Velocity)** 벡터가 필수적입니다.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Ball:
    name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 3D 각속도 [wx, wy, wz] (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    state: BallState = BallState.STATIONARY
    radius: float = 0.03275  # 65.5mm 당구공의 반경
    mass: float = 0.21       # 210g
```

---

## 3. 핵심 물리량: 접촉점의 슬립 속도 ($v_c$)

공이 바닥과 맞닿은 지점(Contact Point)이 바닥에 대해 상대적으로 움직이는 속도를 **슬립 속도($\vec{v}_c$)**라고 합니다.

$$\vec{v}_c = \vec{v}_{cm} + \vec{\omega} \times \vec{r}$$

* $\vec{v}_{cm}$: 공 중심의 선속도 `[vx, 0, vz]`
* $\vec{\omega}$: 공의 각속도 `[wx, wy, wz]`
* $\vec{r}$: 중심에서 접촉점까지의 벡터 `[0, -R, 0]`

이 벡터 외적을 2D 평면(X-Z) 성분으로 풀어서 최적화한 코드가 바로 아래와 같습니다.

```python
# physics.py 내부 (_apply_floor_friction 메서드)
R = ball.radius
vx, _, vz = ball.velocity
wx, wy, wz = ball.angular_velocity

# 외적 정리 결과: 접촉점에서의 미끄러짐 속도
vc_x = vx + wz * R
vc_z = vz - wx * R
```

---

## 4. 미끄러짐 마찰 (Sliding Friction)

만약 $\vec{v}_c 
eq 0$ 이라면, 공은 바닥을 긁고 있는 상태(`SLIDING`)입니다.
바닥은 이 미끄러짐을 방해하는 방향으로 **쿨롱 마찰력(Coulomb Friction)**을 가합니다.

$$\vec{F}_f = -\mu_{slide} \cdot m \cdot g \cdot \frac{\vec{v}_c}{|\vec{v}_c|}$$

이 마찰력은 두 가지 일을 합니다.
1. **선속도 감소:** 이동 속도를 늦춥니다. ($a = \frac{F}{m}$)
2. **각속도 증가:** 바닥에서 토크($\tau = r \times F$)를 발생시켜 회전을 만듭니다.

---

## 5. 코드로 보는 마찰과 가속도 (회전 관성)

수식을 코드로 옮기면 다음과 같습니다. 강체의 관성 모멘트 $I = \frac{2}{5}mR^2$ 이므로, 각가속도 $\alpha = \frac{\tau}{I} = \frac{5}{2} \frac{a}{R}$ 가 됩니다. (코드의 `2.5`가 여기서 나옵니다!)

```python
vc_mag = math.hypot(vc_x, vc_z)
if vc_mag > CONTACT_VELOCITY_THRESHOLD:
    ball.state = BallState.SLIDING
    
    # 방향 벡터
    dir_x, dir_z = vc_x / vc_mag, vc_z / vc_mag
    
    # 1. 선가속도 (감속)
    a_mag = MU_SLIDE * GRAVITY
    ball.velocity[0] -= dir_x * a_mag * dt
    ball.velocity[2] -= dir_z * a_mag * dt
    
    # 2. 각가속도 (토크 발생 -> 회전 증가)
    alpha_mag = 2.5 * a_mag / R
    ball.angular_velocity[0] -= dir_z * alpha_mag * dt  # Z축 마찰이 X축 회전 생성
    ball.angular_velocity[2] += dir_x * alpha_mag * dt  # X축 마찰이 Z축 회전 생성
```

---

## 6. 구름 상태로의 전이 (Snap to Rolling)

마찰력이 계속 작용하다 보면, 어느 순간 선속도와 회전 속도가 일치하여 **접촉점 슬립 속도($v_c$)가 0**이 됩니다.
이때 공은 `SLIDING`을 멈추고 `ROLLING` 상태로 전환됩니다.

```python
# physics.py: _snap_to_rolling 로직
if vc_mag <= CONTACT_VELOCITY_THRESHOLD:
    ball.state = BallState.ROLLING
    
    # 수치 오차 보정: 완벽한 Rolling 상태로 강제 동기화 (v = r * w)
    vx, _, vz = ball.velocity
    ball.angular_velocity[0] = vz / R
    ball.angular_velocity[2] = -vx / R
```

> **🔥 [실무 팁] 수치 적분의 한계 극복**
> 컴퓨터는 이산적인 시간(`dt`) 단위로 계산하기 때문에 $v_c$가 정확히 `0.0`이 되는 순간을 맞추기 힘듭니다. 따라서 임계값(`THRESHOLD`) 이하로 떨어지면 상태를 강제로 전이(Snap)시키고 값을 맞춰주는 것이 물리 엔진 설계의 핵심입니다.

---

## 7. 구름 마찰 (Rolling Friction)

일단 구르기 시작(`ROLLING`)하면, 더 이상 미끄러짐 마찰 계수(`MU_SLIDE`)가 아닌, 훨씬 작은 구름 마찰 계수(`MU_ROLL`)의 지배를 받으며 서서히 정지합니다.

```python
elif ball.state == BallState.ROLLING:
    speed = math.hypot(vx, vz)
    if speed > VELOCITY_THRESHOLD:
        # 구름 마찰력 적용 (선속도 감속)
        a_roll = MU_ROLL * GRAVITY
        drop = a_roll * dt
        
        # ... 속도 감속 처리 로직 ...
        
        # 선속도에 맞추어 각속도도 자동 갱신
        ball.angular_velocity[0] = ball.velocity[2] / R
        ball.angular_velocity[2] = -ball.velocity[0] / R
```

---

## 💡 1교시 요약 및 다음 시간 예고

* **강체 역학의 핵심:** 당구 시뮬레이션은 위치(`position`)만 갱신하는 것이 아니라, 회전(`angular_velocity`)이 이동을 지배하는 세계입니다.
* **상태 전이:** 공은 타격 직후 미끄러지다가(`SLIDING`), 마찰에 의해 회전이 걸리면서 완벽한 구름 상태(`ROLLING`)로 전이합니다.
* **코딩 스킬:** 복잡한 3D 벡터 외적과 관성 모멘트 공식들이 파이썬 코드 몇 줄로 어떻게 아름답게 정리되는지 확인했습니다.

**[다음 시간 예고: 2교시]**
👉 공과 공은 어떻게 충돌하는가? (에너지 보존 법칙)
👉 왜 쿠션을 맞으면 이상한 회전(역회전)이 걸리는가? (`RAIL_H_OFFSET`의 비밀)
