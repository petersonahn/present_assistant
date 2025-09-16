# 🎯 실시간 면접 피드백 시스템

AI 기반 실시간 포즈 분석을 통한 면접 피드백 시스템입니다.

## 📁 프로젝트 구조

```
InterviewBuddy/
├── services/
│   ├── v3-pose/           # 포즈 분석 서비스
│   │   ├── main.py        # FastAPI 서버
│   │   ├── pose_estimator.py  # 포즈 추정 모듈
│   │   ├── requirements.txt   # Python 의존성
│   │   ├── run.sh        # 실행 스크립트
│   │   └── pose/         # OpenVINO 모델 파일
│   │       ├── human-pose-estimation-0001.xml
│   │       └── human-pose-estimation-0001.bin
│   └── v3-speech/         # 🎤 실시간 음성 분석 서비스 (NEW!)
│       ├── main.py        # FastAPI 서버
│       ├── requirements.txt   # Python 의존성 (오디오 라이브러리 포함)
│       ├── Dockerfile     # Docker 설정 (오디오 지원)
│       ├── run.sh         # 실행 스크립트
│       ├── test_client.py # 테스트 클라이언트
│       ├── performance_test.py # 성능 테스트 도구
│       ├── config/        # 설정 파일들
│       │   └── audio_config.py # 오디오 처리 설정
│       └── speech/        # 음성 처리 모듈들
│           ├── audio_capture.py    # sounddevice 실시간 캡처
│           ├── audio_analyzer.py   # librosa 음성 특징 분석
│           ├── speech_recognizer.py # Whisper 음성-텍스트 변환
│           ├── emotion_detector.py  # 감정 분석 (Hugging Face)
│           ├── speech_service.py    # 통합 음성 분석 서비스
│           └── error_handler.py     # 에러 처리 및 안정성
├── frontend/             # 웹 프론트엔드
│   ├── public/
│   │   └── index.html    # 메인 웹 페이지
│   └── static/
│       ├── style.css     # 스타일시트
│       ├── script.js     # JavaScript
│       └── ui_components.svg
├── gateway/              # API 게이트웨이
└── db/                   # 데이터베이스
```

## 🚀 실행 방법

### 1. 시스템 요구사항

- Docker & Docker Compose
- WSL2 (Windows) 또는 Linux/macOS

### 2. 환경 설정

#### .env 파일 생성
프로젝트 루트에 `.env` 파일을 생성하세요:

```bash
# 실시간 면접 피드백 시스템 환경 변수

# 포트 설정
GATEWAY_PORT=15000
FACE_PORT=15010
EMOTION_PORT=15011
POSE_PORT=15012
PROSODY_PORT=15020
FEEDBACK_PORT=15030
FRONTEND_PORT=3000

# MySQL 설정
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DB=interview_db
MYSQL_USER=interview_user
MYSQL_PASSWORD=interview_pass
MYSQL_PORT=3306

# MongoDB 설정
MONGO_INITDB_ROOT_USERNAME=admin
MONGO_INITDB_ROOT_PASSWORD=adminpassword
MONGO_DB=interview_mongo
MONGO_PORT=27017
```

### 3. Docker로 실행

#### 전체 시스템 실행 (권장)

```bash
# 모든 서비스 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build
```

#### v3-pose 서비스만 실행

```bash
# v3-pose 서비스만 빌드 및 실행
docker-compose up --build v3-pose

# 의존성이 있는 서비스들과 함께 실행
docker-compose up --build v3-pose mysql mongo
```

#### v3-speech 서비스만 실행

```bash
# v3-speech 서비스만 빌드 및 실행
docker-compose up --build v3-speech

# 의존성이 있는 서비스들과 함께 실행
docker-compose up --build v3-speech mysql mongo

# GPU 지원 환경에서 실행 (선택사항)
USE_GPU=true docker-compose up --build v3-speech
```

### 4. 개발 모드 실행

```bash
# 개발 컨테이너로 접속
docker-compose up -d dev
docker-compose exec dev bash

# 컨테이너 내에서 직접 실행
cd services/v3-pose
python3 main.py
```

### 5. 접속

- **웹 인터페이스**: http://localhost:3000 (Frontend)
- **API 게이트웨이**: http://localhost:15000
- **v3-pose API**: http://localhost:15012
  - **API 문서**: http://localhost:15012/docs
  - **헬스체크**: http://localhost:15012/health
- **v3-speech API**: http://localhost:15013 🎤
  - **API 문서**: http://localhost:15013/docs
  - **헬스체크**: http://localhost:15013/health
  - **실시간 스트리밍**: http://localhost:15013/speech/stream

## 🎮 사용법

1. 웹 브라우저에서 http://localhost:8000 접속
2. "시작" 버튼을 클릭하여 카메라 권한 허용
3. 실시간 포즈 분석 및 피드백 확인
4. 설정 버튼으로 분석 간격, 민감도 등 조정 가능

## 📊 주요 기능

### 포즈 분석 (v3-pose)
- **어깨 균형**: 좌우 어깨의 수평 정렬 상태
- **머리 위치**: 목과 머리의 정렬 상태  
- **팔 자세**: 팔의 자연스러운 위치

### 🎤 실시간 음성 분석 (v3-speech) - NEW!
- **음성 인식**: Whisper 모델로 실시간 음성-텍스트 변환
- **음성 품질 분석**: 말하기 속도, 명료도, 볼륨 측정
- **감정 분석**: 한국어 특화 감정 상태 분석
- **면접 피드백**: 긴장도, 자신감, 스트레스 지표 제공
- **다국어 지원**: 한국어/영어 자동 감지

### 실시간 피드백
- 종합 점수 (0-100점)
- 구체적인 개선 제안
- 시각적 키포인트 표시
- 음성 품질 실시간 모니터링

### 설정 옵션
- 분석 간격 조정 (1-5초)
- 키포인트 표시/숨김
- 민감도 조절
- 오디오 디바이스 선택
- GPU/CPU 처리 모드 전환

## 🔧 API 엔드포인트

### v3-pose 서비스
- `GET /`: 메인 웹 페이지
- `GET /health`: 서버 상태 확인
- `POST /pose/analyze`: 이미지 파일 포즈 분석
- `POST /pose/analyze_base64`: Base64 이미지 포즈 분석
- `GET /pose/keypoints`: 키포인트 정보
- `POST /pose/feedback`: 키포인트 기반 피드백 생성

### 🎤 v3-speech 서비스 - NEW!
- `GET /`: 메인 웹 페이지
- `GET /health`: 서버 상태 확인
- `POST /speech/start_realtime`: 실시간 음성 분석 시작
- `POST /speech/stop_realtime`: 실시간 음성 분석 중지
- `GET /speech/status`: 현재 분석 상태 조회
- `GET /speech/results/latest`: 최신 분석 결과 조회
- `GET /speech/summary`: 세션 요약 정보
- `GET /speech/stream`: 실시간 결과 스트리밍 (SSE)
- `GET /audio/devices`: 오디오 디바이스 목록
- `POST /audio/test_device/{id}`: 디바이스 테스트

## 🛠️ 개발 정보

### 기술 스택
- **Backend**: FastAPI, OpenVINO, OpenCV
- **Frontend**: Vanilla JavaScript, CSS3
- **포즈 분석**: Intel OpenVINO Human Pose Estimation
- **🎤 음성 처리**: sounddevice, librosa, OpenAI Whisper
- **🤖 AI 모델**: Hugging Face Transformers (감정 분석)
- **🔊 실시간 처리**: 멀티스레딩, 비동기 처리

### 모델 정보
- **모델**: human-pose-estimation-0001
- **키포인트**: 18개 관절점 (COCO 포맷)
- **추론 디바이스**: CPU (기본값)

## 🐛 문제 해결

### 일반적인 문제

1. **모듈을 찾을 수 없음**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **카메라 접근 권한**
   - 브라우저에서 카메라 권한 허용
   - HTTPS 환경에서 실행 권장

3. **모델 파일 없음**
   - `pose/` 디렉토리에 OpenVINO 모델 파일 확인
   - `.xml`, `.bin` 파일 모두 필요

4. **포트 충돌**
   - 8000번 포트가 사용 중인 경우 `main.py`에서 포트 변경

### 로그 확인
```bash
# 서버 실행 시 콘솔에서 로그 확인
python3 main.py

# 백그라운드 실행 시
nohup python3 main.py > server.log 2>&1 &
tail -f server.log
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
