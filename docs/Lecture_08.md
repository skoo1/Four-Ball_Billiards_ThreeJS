---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 3. 3D 가시화와 웹 통신
**8교시: 웹 소켓(WebSocket) 기반 실시간 동기화 기법**

---

## 🎯 이번 시간의 목표

파이썬 물리 엔진(Layer 1, 2)과 브라우저(Layer 3)를 실시간으로 연결합니다.

1. **HTTP vs WebSocket:** 왜 시뮬레이터에는 HTTP REST API를 쓰면 안 되는가?
2. **FastAPI 웹 소켓 연동:** 서버-클라이언트 간의 양방향 통신 채널 열기.
3. **비동기 게임 루프 (`asyncio`):** 1초에 60번(60 FPS) 물리 연산 구동하기.
4. **상태 직렬화 (Serialization):** 3D 좌표와 쿼터니언을 JSON으로 압축하여 전송하기.

---

## 1. 한계점: 왜 HTTP가 아닐까?

일반적인 웹사이트는 클라이언트가 요청(Request)하면 서버가 응답(Response)하고 연결을 끊어버리는 **HTTP 프로토콜(Stateless)**을 사용합니다.

* **HTTP로 당구 게임을 만든다면?**
  * 브라우저: "공 지금 어디 있어?" -> 서버: "x=1, y=2야" (연결 끊김)
  * 브라우저: "지금은 어디 있어?" -> 서버: "x=1.1, y=2.1이야" (연결 끊김)
  * 1초에 60번씩 통신 채널을 열고 닫으면 엄청난 네트워크 오버헤드가 발생하여 화면이 뚝뚝 끊깁니다.

**[해결책: WebSocket]**
한 번 파이프라인(연결)을 뚫어놓고, 양쪽에서 원할 때마다 자유롭게 데이터를 쏘아대는(Full-Duplex) **웹 소켓(WebSocket)**을 사용해야 합니다.

---

## 2. FastAPI를 이용한 서버 셋업

최신 파이썬 웹 프레임워크인 `FastAPI`는 비동기 처리(`ASGI`)와 웹 소켓을 기본으로 지원합니다. 클라이언트가 접속하면 소켓을 `accept()` 하고 목록에 저장합니다.

```python
# server.py 내부
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

clients: list[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()  # 클라이언트 연결 수락
    clients.append(ws) # 활성 클라이언트 목록에 추가
    try:
        while True:
            msg = await ws.receive_text() # 클라이언트의 키보드 입력 수신
            process_input(msg)
    except WebSocketDisconnect:
        clients.remove(ws) # 연결이 끊어지면 목록에서 제거
```

---

## 3. 비동기 게임 루프 (Async Game Loop)

웹 서버는 클라이언트 요청을 기다리면서도, 백그라운드에서는 쉬지 않고 물리 연산을 돌려야 합니다. 이를 위해 `asyncio.sleep`을 활용한 **비동기 게임 루프**를 만듭니다.

```python
async def game_loop():
    dt = 0.001       # 1밀리초 단위의 정밀한 물리 적분 시간
    substeps = 4     # 1프레임당 4번의 물리 연산 수행
    
    while True:
        # 1. 물리 엔진 업데이트
        for _ in range(substeps):
            ctrl.step(dt)
            
        # 2. 결과 브로드캐스트
        await broadcast_state()
        
        # 3. 60 FPS를 맞추기 위한 대기 (다른 비동기 작업에 CPU 양보)
        await asyncio.sleep(1 / 60) 
```

---

## 4. 상태 직렬화 (Serialization)와 데이터 압축

물리 엔진의 `Ball` 객체는 파이썬 메모리 속에 있습니다. 이를 브라우저의 Javascript가 이해할 수 있도록 **JSON 텍스트**로 변환(직렬화)해야 합니다.

```python
# 서버에서 브라우저로 쏘는 매 프레임 데이터 구조 (JSON)
{
  "type": "frame",
  "balls": [
    {
      "name": "white",
      "pos": [0.523, 0.032, -0.115],      # [X, Y, Z] 위치 벡터
      "w": [12.4, -0.5, 3.1],             # [w_x, w_y, w_z] 각속도 벡터
      "state": "ROLLING"
    },
    // ... 다른 공들 데이터 ...
  ]
}
```
> **🔥 [실무 팁] 네트워크 대역폭 최적화**
> 초당 60번씩 데이터를 쏘기 때문에 JSON의 크기를 줄이는 것이 생명입니다. 변수명을 짧게(`position` -> `pos`)하고 소수점 자릿수를 반올림(`round()`)하여 용량을 최소화해야 끊김 없는 렌더링이 가능합니다.

---

## 5. 서버 수명 주기 관리 (Lifespan)

게임 루프는 영원히 도는 `while True` 문입니다. 서버가 켜질 때 이 루프를 시작하고, 서버가 꺼질 때 안전하게 종료(Cancel)하려면 **Context Manager**를 사용해야 합니다.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시: 백그라운드 게임 루프 태스크 생성
    task = asyncio.create_task(game_loop())
    yield  # 서버 실행 중...
    # 서버 종료 시: 게임 루프 강제 종료
    task.cancel()

app = FastAPI(lifespan=lifespan)
```
이 구조가 없으면 서버를 껐는데도 백그라운드에서 물리 엔진이 계속 돌아가는 '좀비 프로세스'가 발생합니다.

---

## 💡 8교시 요약 및 다음 시간 예고

* **웹 소켓(WebSocket):** 실시간 게임 통신을 위한 필수 프로토콜. 양방향으로 지연 없이 데이터를 쏠 수 있습니다.
* **비동기 프로그래밍:** `asyncio`를 이용해 서버가 연결 요청을 받는 동시에 백그라운드에서 물리 연산을 수행하게 만듭니다.
* **직렬화 최적화:** 네트워크로 전송되는 JSON 프레임의 크기를 최소화하여 60 FPS 환경에서의 트래픽 부하를 줄여야 합니다.

**[다음 시간 예고: 9교시]**
👉 파이썬 서버가 보내준 JSON 좌표, 화면에 어떻게 띄울까?
👉 Three.js를 이용한 당구대 렌더링과 카메라 시점 제어 기법!
