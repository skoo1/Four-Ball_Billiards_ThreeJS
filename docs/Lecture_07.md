---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 3. 3D 가시화와 웹 통신
**7교시: 3D 그래픽스 수학 (오일러 각도와 Quaternion의 이해)**

---

## 🎯 이번 시간의 목표

물리 엔진이 계산한 각속도(Angular Velocity)를 이용해 3D 화면 속의 당구공을 '자연스럽게' 굴려봅시다.

1. **오일러 각도(Euler Angle)의 한계:** 짐벌 락(Gimbal Lock) 현상 이해하기.
2. **쿼터니언(Quaternion):** 3D 회전을 수학적으로 가장 완벽하게 표현하는 방법.
3. **회전 누적 (Accumulation):** 매 프레임 발생하는 미세한 회전을 누적하기.
4. **🚨 실무 디버깅 사례:** 3D 엔진마다 다른 행렬 곱셈 순서(Panda3D vs Three.js)로 인해 발생했던 '축 꼬임' 버그 완벽 분석.

---

## 1. 직관의 함정: 오일러 각도 (Euler Angles)

3D 회전을 표현하는 가장 직관적인 방법은 X, Y, Z 세 축을 기준으로 얼마나 회전했는지 (Pitch, Yaw, Roll) 각도로 저장하는 것입니다.
* 예: `rotation = (45, 90, 0)`

**[문제점: 짐벌 락 (Gimbal Lock)]**
두 축이 겹쳐버리면(예: Y축으로 90도 회전 시 X축과 Z축이 일치해버림), 한 축의 회전 자유도를 잃어버리는 **짐벌 락** 현상이 발생합니다. 
쿠션을 맞고 사방으로 미친 듯이 스핀을 먹는 당구공을 오일러 각도로 렌더링하면, 어느 순간 공이 갑자기 홱 뒤집히거나 비정상적으로 덜덜 떨리게 됩니다.

---

## 2. 구원자: 쿼터니언 (Quaternion, 사원수)

3D 그래픽스에서는 짐벌 락을 피하기 위해 복소수를 4차원으로 확장한 **쿼터니언(Quaternion)**을 사용합니다.
$$ q = w + xi + yj + zk $$

복잡한 허수 수학을 다 알 필요는 없습니다. 실무에서는 **"어떤 축(Axis)을 기준으로, 얼마나(Angle) 회전할 것인가?"** 만 알면 쿼터니언을 만들 수 있습니다.

```python
# 회전축(Axis)과 회전각(Angle)을 이용한 쿼터니언 생성
axis = [0, 1, 0] # Y축 기준
angle = 0.05     # 0.05 라디안 회전
dq = Quaternion(axis, angle) 
```

---

## 3. 회전의 누적 (Accumulating Rotation)

당구공은 매 프레임마다 이전 회전 상태에 새로운 회전을 더해야 합니다.
쿼터니언에서는 두 회전을 더할 때 '덧셈'이 아니라 **'곱셈(Multiplication)'**을 사용합니다.

$$ q_{new} = \Delta q \times q_{old} $$

물리 엔진의 각속도($\vec{\omega}$)를 이용하여 이번 프레임의 회전량($\Delta q$)을 구하는 공식:
* **회전 속력 (Angle):** $angle = |\vec{\omega}| \times \Delta t$
* **회전 축 (Axis):** $axis = \frac{\vec{\omega}}{|\vec{\omega}|}$

---

## 4. 🚨 실무 디버깅 사례: 왜 공이 바퀴 빠진 것처럼 돌까?

우리가 초기에 Ursina(Panda3D 기반)로 렌더러를 짤 때, 공이 쿠션을 몇 번 맞고 나면 축이 꼬여서 바퀴 빠진 자동차처럼 비스듬하게 회전하는 기괴한 버그가 있었습니다.

**[원인: 수학적 기준 프레임의 차이]**
새로운 회전(`dq`)과 기존 회전(`q_old`)을 곱할 때, 곱셈의 순서가 매우 중요합니다!
* `dq * q_old` : 글로벌(World) 축을 기준으로 새로운 회전을 적용.
* `q_old * dq` : 로컬(Local) 축을 기준으로 새로운 회전을 적용.

당구공은 이미 굴러가면서 자신의 축(Local)이 돌아가 있으므로, 무조건 **글로벌(World) 축을 기준으로 회전을 누적**해야 합니다.

---

## 5. 3D 엔진마다 다른 쿼터니언 곱셈 규칙

하지만 놀랍게도 3D 엔진마다 수학 규칙(행렬 기반)이 반대입니다!

1. **Panda3D / Ursina (행 벡터 기준):** 
   * `q_old * dq` 가 글로벌(World) 회전을 의미합니다. (왼쪽에서 오른쪽으로 적용)
   * (우리는 처음에 이론만 보고 `dq * q_old`로 짰다가 로컬 회전이 먹어버린 버그가 났었죠!)
2. **Three.js / Unity (열 벡터 기준):**
   * `dq * q_old` 가 글로벌(World) 회전을 의미합니다. (오른쪽에서 왼쪽으로 적용)

---

## 6. 코드로 보는 Three.js 쿼터니언 렌더링

우리가 최종 구축한 웹 프론트엔드(`app.js`)의 Three.js 렌더링 코드입니다.

```javascript
// app.js 내부 로직
const dt = SIM_DT * SIM_SUBSTEPS;
const wMag = Math.hypot(wx, wy, wz); // 각속도 크기

if (wMag > 1e-6) {
    const angle = wMag * dt;
    const axis = new THREE.Vector3(wx / wMag, wy / wMag, wz / wMag);
    
    // 1. 이번 프레임의 회전량(델타 쿼터니언) 생성
    const dq = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    
    // 2. Three.js의 글로벌 회전 누적 공식 (dq * q_old)
    mesh.quaternion.premultiply(dq); 
}
```

---

## 💡 7교시 요약 및 다음 시간 예고

* **짐벌 락 방지:** 3D 공간에서 연속적인 회전을 누적할 때는 무조건 오일러 각도가 아닌 **쿼터니언(Quaternion)**을 사용해야 합니다.
* **Axis-Angle:** 복잡한 쿼터니언 수학을 몰라도, 회전 축(Axis)과 각도(Angle)만 있으면 쉽게 회전량을 변환할 수 있습니다.
* **곱셈 순서 주의:** 사용하는 3D 엔진(Three.js, Unity, Unreal, Panda3D)의 수학 컨벤션(Row-major vs Column-major)에 따라 쿼터니언 곱셈 순서가 뒤집힐 수 있음을 명심해야 합니다.

**[다음 시간 예고: 8교시]**
👉 브라우저와 파이썬은 어떻게 1초에 60번씩 대화를 나눌까?
👉 FastAPI와 WebSocket을 이용한 실시간(Real-time) 물리 동기화 기법!
