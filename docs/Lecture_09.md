---
marp: true
theme: default
paginate: true
backgroundColor: "#f8f9fa"
---

# 🎱 당구 물리 시뮬레이션 엔진 개발
## Part 3. 3D 가시화와 웹 통신
**9교시: Three.js를 이용한 당구대 렌더링과 3D 시각화**

---

## 🎯 이번 시간의 목표

파이썬 물리 엔진이 계산한 1초에 60번의 데이터를 웹 브라우저의 3D 공간에 아름답게 그려냅니다.

1. **Three.js 기초:** Scene, Camera, Renderer 설정과 OrbitControls.
2. **Procedural Texture:** 외부 이미지 파일 없이 코드로 당구공 무늬(점) 만들기.
3. **데이터 매핑:** 웹 소켓으로 받은 JSON 데이터를 3D 객체(Mesh)의 좌표와 회전값에 적용하기.

---

## 1. Three.js의 3대 요소 (Scene, Camera, Renderer)

3D 그래픽을 브라우저에 띄우려면 무대(Scene), 카메라(Camera), 그리고 화면에 그려주는 화가(Renderer)가 필요합니다.

```javascript
// app.js 내부: Three.js 초기 세팅
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();

// 원근 카메라 설정 (시야각 60도)
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 2.5, 0); // 당구대를 위에서 내려다보는 탑뷰

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 마우스 드래그로 시점을 360도 돌려볼 수 있게 해주는 마법의 컨트롤러!
const controls = new OrbitControls(camera, renderer.domElement);
```

---

## 2. 당구대와 쿠션 렌더링

초록색 바닥(Plane)과 4개의 갈색 쿠션(Box)을 배치하여 당구대를 만듭니다.

```javascript
// 바닥 (녹색 천)
const floorGeo = new THREE.PlaneGeometry(TABLE_WIDTH, TABLE_LENGTH);
const floorMat = new THREE.MeshStandardMaterial({ color: 0x2d6b3f });
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2; // 평면을 눕힘
scene.add(floor);

// 쿠션 (갈색 테두리)
const cushionMat = new THREE.MeshStandardMaterial({ color: 0x6b3a1f });
const topCushion = new THREE.Mesh(
    new THREE.BoxGeometry(TABLE_WIDTH + 0.08, 0.03, 0.04), 
    cushionMat
);
topCushion.position.set(0, 0.015, TABLE_LENGTH / 2 + 0.02);
scene.add(topCushion);
// ... 하단, 좌우 쿠션도 동일하게 배치
```

---

## 3. 💡 당구공 렌더링의 핵심: 절차적 텍스처 (Procedural Texture)

당구공은 완벽한 구형(Sphere)입니다. 만약 공에 아무 무늬가 없다면, **공이 미끄러지는지(Sliding) 구르는지(Rolling) 눈으로 구별할 수 없습니다!**
우리는 외부 이미지 파일을 다운로드하는 대신, 브라우저의 `Canvas API`를 이용해 코드로 직접 점(Dot) 무늬를 생성하여 공에 입힙니다.

```javascript
// app.js: 코드로 텍스처 이미지 그리기
function createDotTexture(baseColorHex) {
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 256;
    const ctx = canvas.getContext('2d');
    
    // 1. 바탕색 칠하기
    ctx.fillStyle = baseColorHex;
    ctx.fillRect(0, 0, 256, 256);
    
    // 2. 무작위 위치에 검은색 점(Dot) 30개 찍기
    ctx.fillStyle = '#1e1e1e';
    for (let i = 0; i < 30; i++) {
        ctx.beginPath();
        ctx.arc(Math.random() * 256, Math.random() * 256, 8, 0, Math.PI * 2);
        ctx.fill();
    }
    return new THREE.CanvasTexture(canvas);
}
```

---

## 4. 서버 데이터 동기화 (Position & Quaternion)

웹 소켓을 통해 `{"type": "frame", "balls": [...]}` JSON 데이터가 날아오면, 이 데이터를 3D 객체에 업데이트합니다. **7교시에서 배운 쿼터니언 곱셈 순서**가 여기서 쓰입니다!

```javascript
// 서버에서 받은 ball 상태 데이터: pos=[x,y,z], w=[wx,wy,wz]
const mesh = ballObjects[ballData.name];

// 1. 위치(Position) 업데이트
mesh.position.set(ballData.pos[0], BALL_RADIUS + ballData.pos[1], ballData.pos[2]);

// 2. 회전(Rotation) 업데이트
const dt = SIM_DT * SIM_SUBSTEPS;
const wMag = Math.hypot(ballData.w[0], ballData.w[1], ballData.w[2]);

if (wMag > 1e-6) {
    const angle = wMag * dt;
    const axis = new THREE.Vector3(ballData.w[0]/wMag, ballData.w[1]/wMag, ballData.w[2]/wMag);
    
    // Three.js의 월드 축 기준 회전 누적 공식 (dq * q_old)
    const dq = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    mesh.quaternion.premultiply(dq); 
}
```

---

## 💡 9교시 요약 및 다음 시간 예고

* **OrbitControls:** 3D 시뮬레이터에서 마우스로 시점을 자유자재로 다루게 해주는 강력한 도구.
* **Procedural Texture:** 정적 파일(이미지)에 의존하지 않고 코드로 무늬를 만들어 내어, 시각적으로 회전을 명확하게 표현.
* **데이터 기반 렌더링:** Three.js는 서버(물리 엔진)가 주는 좌표와 쿼터니언 데이터를 그대로 반영하기만 하는 완벽한 L3(View) 역할을 수행합니다.

**[다음 시간 예고: 10교시]**
👉 10시간의 대장정 마무리! 대망의 마지막 수업.
👉 만들어진 3-Tier 아키텍처를 바탕으로, 어떻게 **인공지능(AI) 강화학습 환경(Gymnasium)**을 1시간 만에 구축할 수 있는지 알아봅니다.
