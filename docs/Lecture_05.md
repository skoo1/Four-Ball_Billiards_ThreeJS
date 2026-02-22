---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 2. 소프트웨어 공학과 아키텍처
**5교시: 3-Tier 아키텍처 설계와 시스템 분리 (Decoupling)**

---

## 🎯 이번 시간의 목표

물리 법칙(수학), 게임 규칙(논리), 3D 렌더링(시각화)을 한 파일에 섞어 짜면 어떤 지옥이 펼쳐지는지 이해하고, 이를 우아하게 분리하는 구조를 설계합니다.

1. **스파게티 코드의 함정:** 강한 결합(Tight Coupling)의 문제점.
2. **3-Tier 아키텍처:** L1(물리), L2(컨트롤러), L3(렌더링)의 역할 분담.
3. **이벤트 큐(Event Queue) 패턴:** 모듈 간의 의존성을 끊는 방법.
4. **헤드리스(Headless) 시뮬레이션:** 렌더링 없이 초고속으로 물리 연산만 수행하여 AI 학습 환경 구축하기.

---

## 1. 초보자의 흔한 실수: 스파게티 코드

학생들이 처음 게임이나 시뮬레이터를 만들 때 가장 많이 하는 실수는 **`물리 연산 루프` 안에 `그래픽 렌더링 함수`나 `사운드 재생 코드`를 직접 넣는 것**입니다.

```python
# ❌ 나쁜 예: 강하게 결합된 코드 (Tight Coupling)
def update_physics():
    ball.position += velocity * dt
    if check_collision():
        play_sound("bang.mp3")           # 물리 함수가 사운드 모듈에 의존함!
        draw_effect("sparkle")           # 물리 함수가 그래픽 모듈에 의존함!
        if player1_score == 3:
            show_ui("Player 1 Wins!")    # 물리 함수가 UI 모듈에 의존함!
```

**문제점:** 이렇게 짜면 화면(3D 그래픽) 없이 AI만 1만 번 돌려보고 싶어도, 사운드와 그래픽을 끌 수가 없어서 프로그램이 다운되거나 학습이 불가능해집니다.

---

## 2. 해결책: 3-Tier 아키텍처 (Layered Architecture)

우리는 시스템을 철저하게 3개의 계층(Layer)으로 분리합니다.

* **[Layer 1] Physics Engine (`physics.py`):** 
  * 오직 순수한 파이썬과 수학(Numpy)만 존재합니다. 화면이 어떻게 생겼는지 전혀 모릅니다.
* **[Layer 2] Controller (`controller.py`):** 
  * 턴 관리, 점수 계산, 당구 룰을 담당합니다. L1을 조종하는 뇌(Brain) 역할입니다.
* **[Layer 3] Visualizer (`server.py` + `app.js` 등):** 
  * L2가 계산해 놓은 데이터(`x, y, z`, `회전량`)를 가져다 예쁘게 그리기만 합니다. 3D 엔진(Ursina, Three.js)이 여기에 위치합니다.

---

## 3. 의존성 끊기: 이벤트 큐 (Event Queue) 패턴

L2(컨트롤러)는 공이 부딪혔을 때 사운드를 재생해야 한다는 것을 압니다. 하지만 사운드 라이브러리를 직접 호출하지 않고, **메모장(Queue)에 할 일을 적어두기만 합니다.**

```python
# controller.py 내부 (Layer 2)
class BilliardsController:
    def __init__(self):
        self.pending_events = [] # 렌더링/UI 처리를 위한 메모장
        self.physics_events = [] # 사운드 처리를 위한 메모장

    def execute_command(self, cmd):
        if cmd == "score":
            self.player_score += 1
            # 직접 UI 함수를 호출하지 않고, 이벤트를 발행(Publish)한다.
            self.pending_events.append({"type": "show_result", "msg": "Score!"})
```

---

## 4. Layer 3의 역할: 이벤트 소비 (Consume)

실제 웹 서버나 화면 렌더러(Layer 3)는 매 프레임마다 컨트롤러의 메모장(`pending_events`)을 확인하고, 적혀있는 지시를 수행한 뒤 메모장을 지웁니다.

```python
# server.py 내부 (Layer 3 - 웹 소켓 브로드캐스트)
async def game_loop():
    while True:
        ctrl.step(dt) # 1. 로직 및 물리 업데이트
        
        # 2. 발생한 이벤트(사운드, 점수 등)를 가져와서 클라이언트로 전송
        events_to_send = ctrl.pending_events.copy()
        ctrl.pending_events.clear() # 메모장 비우기
        
        # 3. JSON으로 묶어서 웹 브라우저로 쏘기!
        await broadcast({"balls": ctrl.get_state_json(), "events": events_to_send})
        await asyncio.sleep(1/60)
```
이렇게 하면 L2는 L3가 파이썬이든, 웹 프론트엔드이든 신경 쓸 필요가 없어집니다!

---

## 5. 아키텍처의 꽃: 헤드리스 API (`simulate_shot`)

의존성을 완벽히 끊어낸 덕분에, 우리는 3D 화면을 띄우지 않고 메모리 상에서만 초고속으로 공을 굴려보는 **헤드리스(Headless)** 함수를 만들 수 있게 되었습니다.

```python
# controller.py 내부: 렌더링 없이 시뮬레이션 끝까지 돌리기
def simulate_shot(self, ball_name, azimuth, power):
    # 1. 딥카피로 현재 상태 백업 (AI가 미리 쳐보는 상황)
    snapshot = [self._copy_ball(b) for b in self.physics_balls]
    
    # 2. 큐대로 때리기
    self.engine.apply_cue(target_ball, force=power, direction=dir)
    
    # 3. while문으로 공이 멈출 때까지 초고속 연산 (화면 렌더링 없음!)
    while any(b.is_moving() for b in self.physics_balls):
        self.engine.update(self.physics_balls, dt=0.001)
        
    return {"scored": self.check_score(), "balls": self.physics_balls}
```

---

## 💡 5교시 요약 및 다음 시간 예고

* **강한 결합 금지:** 물리 연산 코드에 `print()`나 오디오 재생 같은 부수 효과(Side-effect)를 넣지 마세요.
* **이벤트 기반 통신:** 모듈 간의 통신은 직접 호출이 아닌 이벤트 큐(Event Queue)를 통해 상태를 전달하는 것이 안전합니다.
* **헤드리스(Headless) 연산:** 아키텍처가 잘 분리된 시뮬레이터는 화면 없이도 실행 가능해야 하며, 이는 인공지능(AI) 강화학습 환경의 핵심 토대가 됩니다.

**[다음 시간 예고: 6교시]**
👉 테스트 주도 개발(TDD): 물리 법칙을 코드로 어떻게 검증할 것인가?
👉 "통과한 테스트가 진짜 정답일까?" (자기 충족적 예언의 위험성)
