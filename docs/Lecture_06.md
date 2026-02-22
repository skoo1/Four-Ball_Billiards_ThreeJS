---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 2. 소프트웨어 공학과 아키텍처
**6교시: 물리 엔진 테스트 주도 개발(TDD)과 디버깅의 함정**

---

## 🎯 이번 시간의 목표

물리 법칙을 코드로 옮겼다면, 그 코드가 수학적으로 완벽한지 어떻게 증명할까요?

1. **테스트의 사각지대 (Blind Spots):** 해피 패스(Happy path) 테스트의 위험성.
2. **자기 충족적 예언 (Self-fulfilling Prophecy):** 버그가 버그를 감싸는 현상.
3. **물리적 정합성 검증 (Physical Assertions):** 에너지 보존 법칙과 대칭성 테스트.
4. **결정론 (Determinism) 테스트:** 강화학습(RL)을 위한 필수 조건.

---

## 1. 🚨 실무 디버깅 사례 A: J자 꺾임 버그는 왜 안 걸렸나?

이전 시간에 다뤘던 '다 멈춰가는 공이 갑자기 J자로 꺾이던 버그'를 기억하시나요?
당시 우리에겐 `test_rolling_friction`이라는 테스트 코드가 있었고, 이 코드는 성공(Pass) 상태였습니다.

**[왜 버그를 잡지 못했을까?]**
```python
# 기존의 느슨했던 테스트 코드 (버그를 통과시킴)
def test_rolling_friction():
    # 공을 굴릴 때 '직진 스핀(Top-spin)'만 주고 테스트함!
    ball.angular_velocity = np.array([30.0, 0.0, 0.0]) # w_y (사이드 스핀) = 0.0
    
    engine._apply_curve_force(ball, dt)
```
테스트 작성자가 엣지 케이스를 고려하지 않고 가장 정직한 상황(`w_y = 0`)만 테스트했기 때문에, 커브 공식에 치명적인 발산 버그가 숨어있었음에도 무사통과했던 것입니다.

---

## 2. 🚨 실무 디버깅 사례 B: 쿠션 폭발 버그

쿠션을 맞았을 때 반발력이 비정상적으로 커져서 회전이 폭증하는 버그도 테스트를 무사히 통과했습니다.

**[왜 버그를 잡지 못했을까?]**
```python
# 느슨한 검증(Assertion)
engine._check_cushion_collisions(ball)

# "어쨌든 탑스핀(음수)이 걸리기만 하면 통과!"
assert ball.angular_velocity[0] < 0  
```
값이 `12.3 rad/s`(정상)이든, `204.0 rad/s`(폭발)이든 0보다 작기만 하면 통과하도록 코드를 짰기 때문입니다. **테스트는 '수치적 임계점'이나 '물리적 한계'를 깐깐하게 검증해야 합니다.**

---

## 3. 물리적 정합성 검증 1: 대칭성 (Symmetry) 테스트

제대로 된 물리 엔진 테스트라면, 우측 쿠션을 맞았을 때와 좌측 쿠션을 맞았을 때 결과가 거울처럼 완벽한 **대칭(Symmetry)**을 이루는지 수학적으로 검증해야 합니다.

```python
def test_cushion_left_right_symmetry():
    # 우측 쿠션 충돌
    b_right = Ball("r", position=[MAX_X - R + 0.001, 0, 0], velocity=[1, 0, 1])
    engine._check_cushion_collisions(b_right)
    
    # 좌측 쿠션 충돌 (거울 세팅)
    b_left = Ball("l", position=[MIN_X + R - 0.001, 0, 0], velocity=[-1, 0, 1])
    engine._check_cushion_collisions(b_left)

    # Z축 속도는 동일하고, X축 속도와 Z축 회전은 부호가 반대여야 완벽한 물리 엔진!
    assert math.isclose(b_right.velocity[2], b_left.velocity[2])
    assert math.isclose(b_right.velocity[0], -b_left.velocity[0])
    assert math.isclose(b_right.angular_velocity[2], -b_left.angular_velocity[2])
```

---

## 4. 물리적 정합성 검증 2: 에너지 보존 (Energy Conservation)

마찰이 없는 환경에서 두 공이 충돌했을 때, 시스템 전체의 운동 에너지(Kinetic Energy)가 증가해서는 안 됩니다.

```python
def test_ball_collision_energy_conservation():
    KE_before = 0.5 * BALL_MASS * np.dot(b1.velocity, b1.velocity)

    PhysicsEngine()._resolve_ball_collision(b1, b2)

    KE_after = (0.5 * BALL_MASS * np.dot(b1.velocity, b1.velocity) + 
                0.5 * BALL_MASS * np.dot(b2.velocity, b2.velocity))

    # 반발 계수(e)가 0.94이므로 에너지는 e^2 비율로 줄어들어야 한다.
    ratio = KE_after / KE_before
    assert 0.80 < ratio <= 1.0, "에너지가 폭발하거나 비정상적으로 소실됨!"
```

---

## 5. 결정론 (Determinism) 테스트

우리가 `controller.py`를 분리하고 `simulate_shot`을 만든 가장 큰 이유는 **인공지능(강화학습) 훈련**을 위해서입니다.
AI 환경에서는 동일한 파워와 각도로 샷을 쏩니다면 **소수점 6자리까지 완벽하게 동일한 결과**가 나와야만 합니다.

```python
def test_headless_determinism():
    ctrl1 = BilliardsController()
    ctrl2 = BilliardsController()
    
    res1 = ctrl1.simulate_shot("white", azimuth=30, power=80)
    res2 = ctrl2.simulate_shot("white", azimuth=30, power=80)
    
    # 두 번의 시뮬레이션 결과가 비트(bit) 수준에서 완전히 동일한지 검증
    np.testing.assert_array_equal(res1["obs"], res2["obs"])
```

---

## 💡 6교시 요약 및 다음 시간 예고

* **해피 패스의 함정:** 물리 시뮬레이터 테스트는 항상 극단적인 엣지 케이스(속도가 0으로 수렴할 때, 강한 사이드 스핀이 들어갔을 때 등)를 공략해야 합니다.
* **물리 법칙 Assert:** 느슨한 부호 검사(`< 0`) 대신, 대칭성, 에너지 보존 법칙, 예측된 수식과 오차범위 내 일치(`math.isclose`)하는지를 검사해야 합니다.
* **결정론 보장:** 강화학습 AI 모델이 수렴하기 위해서는 100% 재현 가능한 결정론(Determinism) 구조가 필수적입니다.

**[다음 시간 예고: Part 3. 3D 가시화와 웹 통신]**
👉 7교시: 왜 오일러 각도(Euler Angle)를 쓰면 팽이가 뒤집힐까? 
👉 3D 그래픽스의 핵심, **쿼터니언(Quaternion)** 수학의 세계로!
