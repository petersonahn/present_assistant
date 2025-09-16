# 🎤 v3-speech 실시간 음성 분석 서비스

AI 기반 실시간 음성 분석 및 면접 피드백 시스템

## 📋 주요 기능

### 🔊 실시간 오디오 처리
- **sounddevice**: 16kHz 모노 실시간 마이크 캡처
- **VAD (Voice Activity Detection)**: 음성 활동 자동 감지
- **노이즈 제거**: 적응형 노이즈 필터링
- **디바이스 관리**: 오디오 디바이스 자동 감지 및 테스트

### 🎯 음성 특징 분석 (librosa)
- **말하기 속도**: WPM (Words Per Minute) 계산
- **음성 품질**: 명료도, 볼륨, 피치 분석
- **스펙트럴 특징**: MFCC, 크로마, 스펙트로그램
- **면접 적합성**: 종합 점수 및 개선 제안

### 🗣️ 음성 인식 (Whisper)
- **실시간 STT**: OpenAI Whisper 모델 사용
- **다국어 지원**: 한국어/영어 자동 감지
- **고품질 변환**: 단어별 타임스탬프 및 신뢰도
- **GPU 가속**: CUDA 지원으로 성능 최적화

### 😊 감정 분석 (Hugging Face)
- **한국어 특화**: KoELECTRA 기반 감정 분석
- **실시간 감정 추적**: 긴장도, 자신감, 스트레스 측정
- **면접 피드백**: 감정 상태 기반 개선 제안
- **다국어 백업**: 다국어 감정 분석 모델 지원

## 🚀 빠른 시작

### Docker로 실행 (권장)

```bash
# 전체 시스템 실행
docker-compose up --build v3-speech

# 의존성과 함께 실행
docker-compose up --build v3-speech mysql mongo

# 백그라운드 실행
docker-compose up -d --build v3-speech
```

### 로컬 개발 환경

```bash
# 의존성 설치
cd services/v3-speech
pip install -r requirements.txt

# 오디오 시스템 패키지 (Ubuntu/Debian)
sudo apt-get install libasound2-dev portaudio19-dev libsndfile1

# 서비스 실행
python main.py
```

## 📡 API 엔드포인트

### 🎤 실시간 분석
```http
POST /speech/start_realtime    # 실시간 분석 시작
POST /speech/stop_realtime     # 실시간 분석 중지
GET  /speech/status            # 현재 상태 조회
GET  /speech/stream            # 실시간 결과 스트리밍 (SSE)
```

### 📊 결과 조회
```http
GET /speech/results/latest     # 최신 분석 결과
GET /speech/summary           # 세션 요약 정보
GET /stats                    # 서비스 통계
```

### 🔧 관리 기능
```http
GET /health                   # 헬스체크
GET /audio/devices           # 오디오 디바이스 목록
POST /audio/test_device/{id} # 디바이스 테스트
GET /config                  # 설정 정보
POST /config/update          # 설정 업데이트
```

## 📈 사용 예제

### Python 클라이언트

```python
import requests
import json

# 서비스 상태 확인
response = requests.get("http://localhost:15013/health")
print(response.json())

# 실시간 분석 시작
requests.post("http://localhost:15013/speech/start_realtime")

# 결과 조회
results = requests.get("http://localhost:15013/speech/results/latest")
print(json.dumps(results.json(), indent=2, ensure_ascii=False))

# 분석 중지
requests.post("http://localhost:15013/speech/stop_realtime")
```

### 실시간 스트리밍

```javascript
// Server-Sent Events로 실시간 결과 수신
const eventSource = new EventSource('http://localhost:15013/speech/stream');

eventSource.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('분석 결과:', result);
    
    // UI 업데이트
    updateSpeechScore(result.overall_score);
    updateEmotionalState(result.emotion);
};
```

## ⚙️ 설정

### 환경 변수

```bash
# GPU 사용 여부
USE_GPU=true

# Whisper 모델 크기
WHISPER_MODEL=small

# 언어 설정
WHISPER_LANGUAGE=auto

# 오디오 설정
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_DURATION=1.0
```

### 설정 파일 (config/audio_config.py)

```python
# 주요 설정값들
SAMPLE_RATE = 16000        # 샘플링 레이트
CHUNK_DURATION = 1.0       # 청크 지속시간
VAD_THRESHOLD = 0.01       # 음성 감지 임계값
WHISPER_MODEL = "small"    # Whisper 모델 크기
```

## 🔧 문제 해결

### 일반적인 문제

1. **오디오 디바이스 접근 불가**
   ```bash
   # Docker에서 오디오 디바이스 권한 확인
   docker-compose logs v3-speech
   
   # 호스트에서 오디오 테스트
   arecord -l
   ```

2. **모델 로딩 실패**
   ```bash
   # 모델 캐시 확인
   ls -la ~/.cache/huggingface/
   ls -la ~/.cache/whisper/
   
   # 수동 모델 다운로드
   python -c "import whisper; whisper.load_model('small')"
   ```

3. **메모리 부족**
   ```bash
   # 작은 모델 사용
   export WHISPER_MODEL=tiny
   
   # GPU 메모리 제한
   export GPU_MEMORY_FRACTION=0.5
   ```

### 로그 확인

```bash
# Docker 로그
docker-compose logs -f v3-speech

# 로컬 실행 로그
python main.py 2>&1 | tee speech.log
```

## 📊 성능 최적화

### GPU 가속

```yaml
# docker-compose.yml에서 GPU 설정
v3-speech:
  build: ./services/v3-speech
  environment:
    - USE_GPU=true
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
```

### 메모리 최적화

```python
# config/audio_config.py
class PerformanceConfig:
    MAX_MEMORY_MB = 2048      # 최대 메모리 사용량
    GARBAGE_COLLECTION = True  # 가비지 컬렉션 활성화
    CACHE_SIZE = 50           # 캐시 크기 제한
```

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   sounddevice   │───▶│   librosa        │───▶│   Whisper       │
│  (Audio Capture)│    │ (Feature Extract)│    │ (Speech-to-Text)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│      VAD        │    │   Audio Analysis │    │ Emotion Analysis│
│ (Voice Activity)│    │   (Interview AI) │    │  (Hugging Face) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌──────────────────────┐
                    │   Speech Service     │
                    │  (Integrated Analysis)│
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │     FastAPI          │
                    │  (REST API + SSE)    │
                    └──────────────────────┘
```

## 🤝 기여하기

1. 이슈 생성 또는 기능 제안
2. 포크 및 브랜치 생성
3. 코드 작성 및 테스트
4. 풀 리퀘스트 제출

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🔗 관련 링크

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [librosa 문서](https://librosa.org/)
- [sounddevice 문서](https://python-sounddevice.readthedocs.io/)
