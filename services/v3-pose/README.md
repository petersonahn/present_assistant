# 실시간 면접 피드백 시스템 - 포즈 감지 모듈

OpenVINO 기반 인간 포즈 추정을 사용한 실시간 면접 피드백 시스템입니다.

## 🚀 주요 기능

- **실시간 포즈 감지**: OpenVINO 최적화된 모델로 빠른 추론
- **자세 분석**: 어깨 균형, 머리 위치, 팔 자세 등 분석
- **실시간 피드백**: 면접 자세에 대한 즉각적인 피드백 제공
- **FastAPI 기반 REST API**: 웹 애플리케이션과 쉬운 통합

## 📋 요구사항

- Python 3.8+
- OpenVINO 2023.2.0
- OpenCV 4.8+
- FastAPI 0.104+

## 🛠️ 설치

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **모델 파일 확인**
프로젝트 루트의 `pose/` 디렉토리에 다음 파일들이 있어야 합니다:
- `human-pose-estimation-0001.xml`
- `human-pose-estimation-0001.bin`

## 🏃‍♂️ 실행

### 1. FastAPI 서버 시작
```bash
python main.py
```
또는
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. API 문서 확인
브라우저에서 `http://localhost:8000/docs` 접속

### 3. 테스트 클라이언트 실행
```bash
python test_client.py
```

## 📡 API 엔드포인트

### 기본 정보
- `GET /` - 루트 엔드포인트
- `GET /health` - 헬스체크
- `GET /pose/keypoints` - 키포인트 정보

### 포즈 분석
- `POST /pose/analyze` - 이미지 파일 업로드로 포즈 분석
- `POST /pose/analyze_base64` - Base64 이미지로 포즈 분석
- `POST /pose/feedback` - 키포인트 데이터로 피드백 생성

## 🎯 사용 예시

### Python 클라이언트
```python
import requests
import base64

# 이미지를 Base64로 인코딩
with open('image.jpg', 'rb') as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

# API 호출
data = {
    "image": f"data:image/jpeg;base64,{image_base64}",
    "include_result_image": True
}

response = requests.post("http://localhost:8000/pose/analyze_base64", json=data)
result = response.json()

print(f"자세 점수: {result['data']['analysis']['posture_score']}/100")
for feedback in result['data']['analysis']['feedback']:
    print(f"- {feedback}")
```

### JavaScript 클라이언트
```javascript
const analyzeImage = async (imageBase64) => {
    const response = await fetch('http://localhost:8000/pose/analyze_base64', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageBase64,
            include_result_image: true
        })
    });
    
    const result = await response.json();
    console.log('자세 점수:', result.data.analysis.posture_score);
    console.log('피드백:', result.data.analysis.feedback);
};
```

## 🔍 분석 결과 구조

```json
{
    "success": true,
    "data": {
        "keypoints": [
            {
                "id": 0,
                "name": "nose",
                "x": 320,
                "y": 240,
                "confidence": 0.85
            }
        ],
        "analysis": {
            "posture_score": 85,
            "shoulder_balance": "balanced",
            "head_position": "straight",
            "arm_position": "natural",
            "feedback": [
                "전반적으로 좋은 자세입니다! 👍",
                "어깨 위치가 균형잡혀 있어요 ✓"
            ]
        },
        "image_shape": [480, 640],
        "keypoint_count": 18
    }
}
```

## 🎨 키포인트 정보

총 18개의 키포인트를 감지합니다:
- **얼굴**: nose, l_eye, r_eye, l_ear, r_ear
- **상체**: neck, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist
- **하체**: l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle

## 🏗️ 프로젝트 구조

```
mini/
├── pose/
│   ├── human-pose-estimation-0001.xml    # OpenVINO 모델 (XML)
│   └── human-pose-estimation-0001.bin    # OpenVINO 모델 (Binary)
├── pose_estimator.py                     # 포즈 추정 핵심 로직
├── main.py                               # FastAPI 서버
├── test_client.py                        # 테스트 클라이언트
├── requirements.txt                      # Python 의존성
├── ui_components.svg                     # UI 디자인
└── README.md                            # 문서
```

## 🔧 커스터마이징

### 임계값 조정
`pose_estimator.py`에서 다음 값들을 조정할 수 있습니다:
- 키포인트 신뢰도 임계값: `max_val > 0.1`
- 어깨 균형 임계값: `shoulder_diff < 20`
- 머리 기울기 임계값: `head_tilt < 30`

### 추가 분석 기능
`analyze_pose()` 메서드에 새로운 분석 로직을 추가할 수 있습니다.

## 🐛 문제 해결

### 모델 로딩 실패
- `pose/` 디렉토리에 모델 파일이 있는지 확인
- OpenVINO가 올바르게 설치되었는지 확인

### 웹캠 접근 실패
- 카메라 권한 확인
- 다른 애플리케이션에서 카메라를 사용 중인지 확인

### API 응답 느림
- CPU 대신 GPU 사용: `device="GPU"`로 변경
- 이미지 크기 줄이기
- 배치 처리 고려

## 📈 성능 최적화

- **GPU 사용**: Intel GPU가 있다면 `device="GPU"` 설정
- **모델 양자화**: INT8 모델 사용으로 이미 최적화됨
- **비동기 처리**: FastAPI의 async/await 활용
- **캐싱**: 결과 캐싱으로 반복 요청 최적화

## 📄 라이선스

MIT License

## 🤝 기여

버그 리포트나 기능 제안은 GitHub Issues를 통해 해주세요.
