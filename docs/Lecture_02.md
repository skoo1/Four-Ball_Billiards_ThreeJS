---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 1. 핵심 물리 엔진 구현
**2교시: 충돌 역학 (Collision Dynamics) 및 쿠션 반발의 비밀**

---

## 🎯 이번 시간의 목표

화면 밖으로 나가지 않게 막는 단순한 '벽'과 '공'을 진짜 당구처럼 움직이게 만듭니다.

1. **공-공 충돌 (Ball-to-Ball):** 운동량 보존과 반발 계수(Restitution)를 이용한 충격량(Impulse) 계산.
2. **공-쿠션 충돌 (Ball-to-Cushion):** 단순 반사를 넘어선 쿠션의 물리적 특성 이해.
3. **쿠션의 비밀 (`RAIL_H_OFFSET`):** 쿠션을 맞은 공이 왜 회전(Top-spin)을 먹고 튀어나오는지 물리적으로 구현하기.

---

## 1. 공-공 충돌 감지 (Collision Detection)

두 공의 중심 사이의 거리 $d$가 두 공의 반지름의 합($2R$)보다 작아지면 충돌한 것입니다.
단, 컴퓨터 시뮬레이션에서는 두 공이 겹친 채로(Penetration) 감지되는 경우가 많으므로 이를 밀어내어 분리하는 작업이 선행되어야 합니다.

```python
# physics.py 내부 (_check_ball_collision)
dist_vec = b1.position - b2.position
dist = np.linalg.norm(dist_vec)

if dist < 2 * BALL_RADIUS:
    # 1. 겹침(Penetration) 해결: 두 공을 충돌 표면으로 밀어냄
    overlap = 2 * BALL_RADIUS - dist
    normal = dist_vec / dist
    
    # 질량이 같으므로 절반씩 이동시켜 겹침 해소
    b1.position += normal * (overlap / 2)
    b2.position -= normal * (overlap / 2)
    return True
```

---

## 2. 충격량(Impulse) 계산 공식

충돌 시 두 공이 주고받는 **충격량($J$)**은 반발 계수($e$)와 상대 속도를 이용하여 구합니다. 당구공은 완전 탄성 충돌에 가까우며, 우리 엔진에서는 $e = 0.94$ (`BALL_RESTITUTION`)를 사용합니다.

$$ J = \frac{-(1 + e) \cdot \vec{v}_{rel} \cdot \vec{n}}{\frac{1}{m_1} + \frac{1}{m_2}} $$

* $\vec{v}_{rel}$: 두 공의 상대 속도 ($\vec{v}_1 - \vec{v}_2$)
* $\vec{n}$: 충돌면의 법선 벡터 (Normal vector)
* $e$: 반발 계수 (`BALL_RESTITUTION`)

---

## 3. 코드로 보는 공-공 충돌 해결

계산된 충격량 $J$를 질량 $m$으로 나누어 각각의 속도 변화량($\Delta v$)으로 적용합니다. 

```python
# physics.py 내부 (_resolve_ball_collision)
rel_vel = b1.velocity - b2.velocity
vel_along_normal = np.dot(rel_vel, normal)

# 멀어지고 있는 상태라면 충돌 무시
if vel_along_normal > 0:
    return

# 충격량(J) 스칼라 계산
j = -(1 + BALL_RESTITUTION) * vel_along_normal
j /= (1 / BALL_MASS + 1 / BALL_MASS)

# 벡터로 변환하여 속도에 반영 (작용-반작용)
impulse = j * normal
b1.velocity += impulse / BALL_MASS
b2.velocity -= impulse / BALL_MASS
```

---

## 4. 쿠션 충돌 (Cushion Collision)

공이 테이블의 경계(X_MIN, X_MAX, Z_MIN, Z_MAX)를 넘어가면 쿠션과 충돌한 것입니다.
가장 기본적인 처리는 수직 방향 속도의 부호를 반대로 뒤집고, 반발 계수(`CUSHION_RESTITUTION` = 0.75)를 곱해 에너지를 감소시키는 것입니다.

```python
# 우측 쿠션(X_MAX) 충돌 예시
if ball.position[0] > self.x_max - BALL_RADIUS:
    # 1. 위치 보정 (벽 밖으로 나간 공을 벽에 붙임)
    ball.position[0] = self.x_max - BALL_RADIUS
    
    # 2. 선속도 반사 및 감쇠
    v_perp = ball.velocity[0]
    ball.velocity[0] = -v_perp * CUSHION_RESTITUTION
```

하지만 이것만으로는 진짜 당구의 움직임을 만들 수 없습니다!

---

## 5. 💡 쿠션의 비밀: 왜 공에 전진 회전이 걸릴까?

당구대의 쿠션(고무 레일)은 공의 중심(적도)보다 **약간 높은 곳**에 위치합니다. 이를 물리 엔진에서는 `RAIL_H_OFFSET` ($h$) 으로 정의합니다.

공이 쿠션에 부딪히면, 반발력이 공의 중심이 아닌 $h$만큼 높은 곳에서 작용하므로 **강력한 회전 토크($\tau$)**가 발생합니다.

$$ \tau = F \times h \quad \Rightarrow \quad \Delta \omega = \frac{\Delta p \cdot h}{I} $$

> **👨‍🏫 Instructor Note:**
> 이 로직이 없으면 뱅크샷(빈쿠션 치기)을 할 때 공이 부자연스럽게 통통 튀어 다닙니다. 레일 높이 오프셋이 강제 탑스핀을 먹여주기 때문에, 쿠션을 맞은 공이 바닥을 촥! 움켜쥐며 아름답게 굴러가게 됩니다.

---

## 6. 쿠션 토크(Torque)의 코드 구현

선속도가 반사될 때의 충격량($\Delta p$)을 이용해 회전 각속도($\Delta \omega$)를 얼마나 변화시킬지 계산하여 `angular_velocity`에 더해줍니다.

```python
# physics.py: 쿠션에 의한 강제 탑스핀 적용 로직
# 1. 충격량에 비례하는 각속도 변화량 계산
delta_w = RAIL_H_OFFSET * (1 + CUSHION_RESTITUTION) * abs(v_perp) * BALL_MASS / INERTIA

# 2. 부딪힌 벽의 방향에 따라 회전축 결정 (Z축 기준 회전)
if hitting_right_cushion:
    ball.angular_velocity[2] -= delta_w  # 우측 벽: Z축 음의 방향 회전 생성
```

이처럼 아주 작은 물리 상수(`RAIL_H_OFFSET = 0.003`) 하나가 시뮬레이터의 **'리얼리티(Reality)'**를 결정짓는 핵심 요소가 됩니다.

---

## 💡 2교시 요약 및 다음 시간 예고

* **공-공 충돌:** 운동량 보존 법칙과 반발 계수를 이용한 3D 벡터 충격량 계산.
* **쿠션 충돌의 디테일:** 벽은 단순히 속도를 튕겨내는 거울이 아니다. 타격점의 높이 차이(`RAIL_H_OFFSET`)가 만들어내는 토크가 당구 특유의 구름을 완성한다.

**[다음 시간 예고: 3교시]**
👉 플레이어는 어떻게 공에 스핀(당점)을 넣는가?
👉 **[주의!]** 3D 벡터 외적(Cross Product)의 순서를 틀렸을 때 벌어지는 대참사 (우리가 직접 겪은 버그 디버깅 사례 공개!)
