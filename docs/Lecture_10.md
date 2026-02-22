---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 4. 확장 및 응용
**10교시: Headless API와 AI 강화학습(RL) 연동 (Grand Finale)**

---

## 🎯 이번 시간의 목표

10시간의 대장정을 마무리하며, 우리가 만든 물리 엔진을 **인공지능(AI)**에게 쥐어줍니다.

1. **Headless 시뮬레이션의 위력:** 3-Tier 아키텍처가 빛을 발하는 순간.
2. **OpenAI Gymnasium:** 강화학습 환경의 표준 인터페이스 이해하기.
3. **상태(Observation)와 행동(Action):** AI의 눈과 손 만들기.
4. **보상 함수(Reward Function):** 4구 당구의 룰을 숫자로 번역하기.

---

## 1. 3-Tier 아키텍처의 진가: Headless API

우리는 5교시에서 물리(L1)와 게임 로직(L2)을 그래픽(L3)에서 완전히 분리했습니다.
덕분에 화면 렌더링(Three.js, 60FPS) 없이, 메모리 상에서 while 문을 돌려 **1초에 수천 번의 샷**을 끝까지 연산해 내는 `simulate_shot` 함수를 갖게 되었습니다.

```python
# controller.py 내부의 궁극의 무기: simulate_shot
result = ctrl.simulate_shot(
    ball_name="white", 
    azimuth=45.0,    # 조준각도
    power=80.0,      # 파워
    tip=[0.5, 0.0]   # 당점 (우회전)
)

print(result["scored"])       # True / False (득점 여부)
print(result["cushion_hits"]) # 3 (쿠션 맞은 횟수)
```
이제 AI는 화면을 볼 필요 없이, 이 API만 무한히 호출하며 학습하면 됩니다!

---

## 2. OpenAI Gymnasium 환경 설계

강화학습 알고리즘(PPO, DQN 등)이 우리 게임을 플레이하려면, 전 세계 표준 인터페이스인 `gym.Env` 클래스 규격에 맞춰 게임을 포장(Wrapper)해야 합니다.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BilliardsEnv(gym.Env):
    def __init__(self):
        self.ctrl = BilliardsController()
        # AI의 행동(Action) 공간과 상태(Observation) 공간을 정의해야 함

    def reset(self, seed=None):
        # 게임 초기화 및 첫 화면(상태) 반환
        
    def step(self, action):
        # AI가 행동을 취하면, 물리 엔진을 돌리고 (다음 상태, 보상, 종료여부) 반환
```

---

## 3. AI의 눈과 손 (Observation & Action Space)

**[AI의 눈: Observation Space]**
카메라 픽셀 렌더링을 이미지로 주면 학습이 너무 느립니다. 대신 공들의 `[X, Z]` 좌표를 직접 배열로 전달합니다.

```python
# 4개의 공 * [x, z] = 8개의 실수형(float32) 데이터
self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
```

**[AI의 손: Action Space]**
AI가 큐를 조작할 수 있는 4가지 연속적인(Continuous) 변수를 정의합니다.

```python
# [조준각도(-1~1), 파워(-1~1), 당점X(-1~1), 당점Y(-1~1)]
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
```

---

## 4. 보상 함수 (Reward Function)의 설계

AI는 오직 **'보상(Reward)'**을 극대화하는 방향으로만 움직입니다. 당구 4구의 룰을 보상 숫자로 번역해야 합니다.

```python
def step(self, action):
    # 1. AI의 [-1, 1] 행동을 실제 물리 값(각도 0~360, 파워 0~100)으로 스케일링
    azimuth = (action[0] + 1.0) * 180.0
    power = (action[1] + 1.0) * 50.0
    
    # 2. 물리 엔진(Headless) 초고속 시뮬레이션!
    res = self.ctrl.simulate_shot("white", azimuth=azimuth, power=power)
    
    # 3. 보상(Reward) 계산 (Carrot and Stick)
    reward = 0.0
    if res["foul"]:      reward -= 5.0   # 파울(흰공이 아무것도 못 맞춤) = 벌점
    elif res["scored"]:  reward += 10.0  # 득점(빨간공 2개 히트) = 큰 상점
    elif res["touched"] > 0: reward += 1.0 # 1개만 맞춤 = 작은 상점(격려)
    
    return self._get_obs(), reward, res["scored"], False, {}
```

---

## 5. 학습 루프 실행 (Training Loop)

이제 Stable-Baselines3와 같은 최첨단 강화학습 라이브러리를 연결하면 끝입니다. 

```python
from stable_baselines3 import PPO

# 1. 방금 만든 당구 환경 생성
env = BilliardsEnv()

# 2. PPO(Proximal Policy Optimization) 딥러닝 모델 생성
model = PPO("MlpPolicy", env, verbose=1)

# 3. AI야, 50만 번 당구를 쳐보면서 스스로 깨달아라! (약 10분 소요)
model.learn(total_timesteps=500_000)

# 4. 학습된 뇌(Model) 저장
model.save("billiards_ai_master")
```
그래픽을 껐기 때문에, 여러분의 노트북에서도 눈 깜짝할 사이에 수만 번의 당구 경기가 시뮬레이션됩니다.

---

## 🎓 10시간의 대장정 마무리 (Wrap-up)

여러분은 지난 10시간 동안 엄청난 것을 해냈습니다.

1. **Math & Physics:** 마찰력, 벡터 외적, 에너지 보존 법칙을 직접 코딩했습니다. (버그도 잡았죠!)
2. **Software Engineering:** 3-Tier로 시스템을 분리하고, TDD로 엣지 케이스를 막는 실무 아키텍처를 경험했습니다.
3. **Web & 3D Graphics:** 쿼터니언을 이해하고 WebSocket과 Three.js로 실시간 동기화를 이뤄냈습니다.
4. **Artificial Intelligence:** 마지막으로 이 모든 것을 RL 환경으로 묶어 AI에게 넘겼습니다.

수학 공식 하나가 거대한 풀스택 시뮬레이션과 AI로 확장되는 이 짜릿한 경험이, 앞으로 여러분의 공학도 인생에 든든한 무기가 되길 바랍니다. 수고하셨습니다!
