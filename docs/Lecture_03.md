---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 1. 핵심 물리 엔진 구현
**3교시: 당점과 스핀 (English & Torque), 그리고 3D 벡터 외적의 함정**

---

## 🎯 이번 시간의 목표

당구의 진정한 묘미는 공의 정중앙이 아닌 다른 곳을 쳐서 회전을 만들어내는 **당점(Tip Offset/English)**에 있습니다.

1. **타격 역학:** 편심 타격(Off-center hit)이 만들어내는 선속도와 각속도.
2. **로컬 좌표계 구축:** 큐가 바라보는 방향을 기준으로 '오른쪽'과 '위쪽' 벡터 구하기.
3. **🚨 실무 디버깅 사례:** 벡터 외적(Cross Product)의 순서를 틀렸을 때 발생하는 대참사와 '자기 충족적 예언'의 위험성.

---

## 1. 편심 타격 (Off-center Hit)의 물리

큐대로 공의 정중앙에서 벗어난 위치를 타격하면, 타격 힘($\vec{F}$)은 두 가지 역할을 합니다.

1. **선속도 생성:** 질량 중심을 밀어냅니다. ($\Delta \vec{v} = \frac{\vec{F} \Delta t}{m}$)
2. **각속도 생성:** 중심에서 벗어난 거리(모멘트 암, $\vec{r}_{tip}$)와 힘이 만나 토크($\vec{\tau}$)를 발생시킵니다.
   $$ \vec{\tau} = \vec{r}_{tip} \times \vec{F} \quad \Rightarrow \quad \Delta \vec{\omega} = \frac{\vec{\tau} \Delta t}{I} $$

---

## 2. 큐의 로컬 좌표계 (Local Coordinate System)

플레이어가 "오른쪽 당점(Tip X = 0.5)을 줘!"라고 명령했을 때, 3D 공간 상에서 그 타격점($\vec{r}_{tip}$)이 어디인지 계산하려면 **큐대 기준의 직교 좌표계**가 필요합니다.

* $\vec{Forward}$: 큐가 조준하는 방향
* $\vec{Up}_{global}$: 월드의 절대 위쪽 `[0, 1, 0]`
* $\vec{Right}$: 큐대 기준 오른쪽
* $\vec{Up}_{local}$: 큐대 기준 위쪽

이 벡터들을 구하기 위해 **벡터 외적(Cross Product)**을 사용합니다.

---

## 3. 🚨 대참사의 시작: 벡터 외적의 순서 오류

3D 그래픽스의 표준인 **오른손 좌표계(Right-hand rule)**에서 '오른쪽' 벡터를 구하는 공식은 다음과 같습니다.

$$ \vec{Right} = \vec{Forward} \times \vec{Up} $$

**[우리가 실제로 겪었던 버그 코드]**
```python
# physics.py 초기 버전 (버그 존재)
global_up = np.array([0.0, 1.0, 0.0])
right = np.cross(global_up, fwd)  # Up x Forward = Left 벡터가 나옴!!
```
순서를 `Up x Forward`로 잘못 작성한 결과, `right` 변수는 이름과 달리 **왼쪽(Left)**을 가리키게 되었습니다.

---

## 4. 나비 효과 (The Butterfly Effect)

이 사소한 수학적 실수 하나가 엄청난 연쇄 작용을 일으켰습니다.

1. 플레이어가 **우회전(Right English, Tip X > 0)**을 입력합니다.
2. 코드는 `왼쪽 벡터 * Tip X`를 계산하여, 실제로는 공의 **왼쪽 면**을 타격합니다.
3. 공의 왼쪽을 맞았으니 공은 **좌회전(반시계 방향)**을 돌며 날아갑니다.

> **결론:** 당점이 100% 반대로 작동하는 기괴한 물리 엔진이 탄생했습니다!

---

## 5. 스쿼트(Squirt) 현상과 '꼼수'의 위험성

당구에는 공의 오른쪽을 강하게 치면, 공이 큐대의 밀림에 의해 **왼쪽으로 미세하게 밀려 나가는 스쿼트(Squirt)** 현상이 있습니다.

당점이 반대로 적용되다 보니, 당연히 스쿼트의 방향도 현실과 반대로 나왔습니다. 
이때 과거의 개발자(혹은 AI)가 공식의 오류를 찾는 대신 **충격적인 꼼수**를 사용합니다.

```python
# 스쿼트 상수(SQUIRT_FACTOR)의 부호를 강제로 음수(-)로 조작해 버림
SQUIRT_FACTOR: float = -0.02  # 원래는 양수(0.02)여야 함
```

---

## 6. Two Wrongs Make a Right (두 개의 오류가 만든 가짜 정답)

수학(외적)이 틀렸는데, 상수(`SQUIRT_FACTOR`)마저 틀리게 조작해 버리자, **마이너스 곱하기 마이너스가 되어 결과값이 정상**처럼 나오기 시작했습니다.

심지어 이 오염된 코드를 기반으로 테스트 코드(Test Suite)를 작성했기 때문에, 컴퓨터는 "모든 테스트 통과(Pass)!"를 외쳤습니다. 이것이 바로 소프트웨어 공학에서 가장 경계해야 할 **자기 충족적 예언(Self-fulfilling prophecy)**입니다.

*해결:* 우리는 결국 화면에 그려진 3D 그래픽을 직접 눈으로 보고, 인간의 물리적 직관을 동원해서야 이 삼중 부호 오류를 찾아내어 수정할 수 있었습니다.

---

## 7. 올바른 코드 구현 (수정 완료)

```python
# physics.py (apply_cue 로직)
fwd = direction.copy()
global_up = np.array([0.0, 1.0, 0.0])

# 1. 완벽한 직교 프레임 구축 (순서 엄수!)
right = np.cross(fwd, global_up)      # Forward x Up = Right
local_up = np.cross(right, fwd)       # Right x Forward = Up

# 2. 3D 타격점 계산
contact_point = right * tip_x + local_up * tip_y + fwd * tip_z

# 3. 토크 생성
torque = np.cross(contact_point, force_vec)
ball.angular_velocity += torque / INERTIA
```

---

## 💡 3교시 요약 및 다음 시간 예고

* **당점의 원리:** 오프센터 타격은 모멘트 암을 형성하여 공에 토크(스핀)를 부여한다.
* **좌표계 수학:** 3D 그래픽스와 물리 엔진에서 벡터 외적의 순서는 생명과도 같다.
* **소프트웨어 공학적 교훈:** 테스트 코드가 통과했다고 맹신하지 마라. 잘못된 근본 논리 위에 세워진 테스트는 거짓 안도감을 줄 뿐이다.

**[다음 시간 예고: Part 2. 소프트웨어 공학과 아키텍처]**
👉 스파게티 코드를 피하는 방법: 물리, 로직, 렌더링을 완전히 분리하는 3-Tier 아키텍처 설계법.
