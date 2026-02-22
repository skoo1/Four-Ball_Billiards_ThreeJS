---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 1. 핵심 물리 엔진 구현
**4교시: 고급 당구 물리 (Curve, Squirt, Throw)와 엣지 케이스 디버깅**

---

## 🎯 이번 시간의 목표

기본적인 마찰과 충돌을 넘어, 당구 시뮬레이션의 '리얼리티'를 극한으로 끌어올리는 고급 물리 현상들을 구현합니다.

1. **커브 / 맛세이 (Curve / Swerve):** 사이드 스핀이 궤적을 휘게 만드는 원리.
2. **🚨 J자 꺾임 (J-Hook) 버그 해결:** 분모가 0에 가까워질 때 발생하는 수치적 폭발(Blow-up) 제어.
3. **스쿼트 (Squirt):** 큐 타격 시 공이 당점 반대 방향으로 밀리는 현상.
4. **스로우 (Throw):** 공과 공이 부딪힐 때 마찰력(`MU_BALL`)에 의한 분리각 변화.

---

## 1. 커브 현상 (Curve / Swerve)

공에 좌우 회전(사이드 스핀, $\omega_y$)을 주고 치면, 공이 미끄러지는 동안(SLIDING) 횡방향(가로 방향) 마찰력이 발생하여 궤적이 포물선처럼 휩니다.

이를 현상론적(Phenomenological) 모델로 근사한 공식은 다음과 같습니다.

$$a_{curve} = \mu_{slide} \cdot g \cdot \left( \frac{\omega_y \cdot R}{v + 0.10} \right)$$

* $\omega_y$: Y축(수직축) 기준 사이드 스핀
* $v$: 현재 공의 전진 속력 (속도가 느려질수록 커브가 급격하게 먹음)
* $+ 0.10$: $v=0$ 일 때 수학적으로 무한대로 발산(0으로 나누기 오류)하는 것을 막기 위한 감쇠항(Damping term).

---

## 2. 🚨 실무 디버깅 사례: 공이 왜 J자로 꺾일까?

초기 물리 엔진에서는 공이 다 굴러가서 멈추기 직전에 갑자기 옆으로 휙! 꺾이는 **J자 꺾임(J-Hook) 버그**가 있었습니다.

**[버그의 원인: 엣지 케이스 누락]**
공식을 다시 보면 분모에 속도 $v$가 있습니다. 공이 거의 멈춰갈 때($v \approx 0$), 마찰로 인해 선속도는 죽었지만 사이드 스핀 $\omega_y$가 약간이라도 남아있다면, 분수가 거대해지면서 **엄청난 횡가속도**가 발생해 버립니다.

---

## 3. 단 한 줄의 조건문으로 물리 법칙 수호하기

물리적으로 **'커브 현상'은 공이 바닥을 긁으며 미끄러질 때(SLIDING)만 발생**해야 합니다.
이미 마찰력을 이겨내고 정상적으로 구르는 상태(ROLLING)에서는 사이드 스핀이 남아있더라도 제자리에서 팽이처럼 돌 뿐, 궤적을 옆으로 휘게 만들지 못합니다.

```python
# physics.py 내부 (_apply_curve_force)

# [버그 픽스] ROLLING 상태에서는 커브 힘을 적용하지 않음!
if ball.state != BallState.SLIDING:
    return

wy = ball.angular_velocity[1] # 사이드 스핀
v_speed = math.hypot(ball.velocity[0], ball.velocity[2])

# ... 커브 가속도 계산 로직 ...
```
상태(State Machine)를 명확히 분리해 둔 덕분에, 이 조건문 하나로 오랜 체증이던 J자 커브 버그를 완벽히 해결할 수 있었습니다.

---

## 4. 스쿼트 현상 (Squirt / Deflection)

공의 오른쪽 측면을 강하게 타격하면, 큐 끝(Tip)이 공을 밀어내는 힘 때문에 공이 내가 조준한 방향보다 **미세하게 왼쪽으로 밀려 나가는 현상**이 발생합니다. 이를 스쿼트(Squirt)라고 합니다.

```python
# physics.py: apply_cue 타격 로직 중 일부

# tip_x: 중심에서 벗어난 좌우 거리 (오른쪽이 +)
# SQUIRT_FACTOR: 스쿼트 계수 (0.02)

if SQUIRT_FACTOR > 0 and abs(tip_x) > 1e-4:
    # 아크탄젠트를 이용해 편향 각도(deflection_angle) 계산
    deflection_angle = -math.atan(tip_x / BALL_RADIUS) * SQUIRT_FACTOR
    
    # 구해진 각도만큼 조준 방향(fwd) 벡터를 강제로 회전시켜 초기 속도에 반영
    sin_d = math.sin(deflection_angle)
    cos_d = math.cos(deflection_angle)
    # ... 2D 회전 변환 행렬 적용 ...
```

---

## 5. 스로우 현상 (Throw / Cut Angle Effect)

두 공이 비스듬하게 충돌할 때, 순수 탄성 충돌이라면 이론적인 분리각으로 벌어져야 합니다.
하지만 두 공의 표면이 비벼지면서 발생하는 **쿨롱 마찰(`MU_BALL`)** 때문에, 타격하는 공의 스핀이 맞는 공에게 기어(Gear)처럼 전달되거나 **분리각이 예상보다 좁아지는 현상**이 발생합니다.

```python
# _resolve_ball_collision 내부: 접촉점의 접선(Tangential) 마찰력 계산
rel_vel = b1.velocity - b2.velocity
tangent_vel = rel_vel - vel_along_normal * normal

if np.linalg.norm(tangent_vel) > 1e-6:
    tangent_dir = tangent_vel / np.linalg.norm(tangent_vel)
    # 마찰 계수(MU_BALL)를 곱한 접선 방향 충격량(Impulse) 적용
    friction_impulse = MU_BALL * j * tangent_dir
    b1.velocity -= friction_impulse / BALL_MASS
    b2.velocity += friction_impulse / BALL_MASS
```
이 로직을 통해 얇게 칠 때(Cut shot) 두께가 미세하게 두꺼워지는 실제 당구의 타격감을 모사합니다.

---

## 💡 4교시 요약 및 Part 1 결론

* **현상론적 모델링:** 수식이 너무 복잡하거나 수치적으로 불안정(0 나누기 등)해질 때, `v + 0.10`과 같은 감쇠항이나 `if state == SLIDING` 같은 상태 제어로 물리적 정합성을 유지합니다.
* **현실의 디테일:** 스쿼트(Squirt)와 스로우(Throw) 같은 고급 물리 효과들이 더해져야 진짜 당구 시뮬레이션으로 인정받을 수 있습니다.
* **축하합니다!** 이것으로 우리는 상용 게임 부럽지 않은 3D 당구 물리 엔진(Layer 1)의 밑바닥부터 옥상까지 모든 구조를 마스터했습니다.

**[다음 시간 예고: Part 2. 소프트웨어 공학과 아키텍처]**
👉 5교시: 물리 엔진을 웹(Web)과 인공지능(AI)에 연결하기 위한 3-Tier 아키텍처 설계법
