#!/bin/bash

# v3-speech 서비스 실행 스크립트

echo "🎤 v3-speech 서비스 시작..."

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:/app"

# 환경 변수 설정
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# GPU 사용 여부 확인
if [ "$USE_GPU" = "true" ]; then
    echo "GPU 모드로 실행"
    export DEVICE="cuda"
else
    echo "CPU 모드로 실행"
    export DEVICE="cpu"
fi

# 오디오 디바이스 확인
echo "오디오 디바이스 확인 중..."
if [ -d "/dev/snd" ]; then
    echo "오디오 디바이스 감지됨"
    ls -la /dev/snd/
else
    echo "경고: 오디오 디바이스를 찾을 수 없습니다"
fi

# 모델 캐시 디렉토리 생성
mkdir -p /root/.cache/huggingface
mkdir -p /root/.cache/whisper

# 서비스 실행
echo "FastAPI 서버 시작 (포트: 15013)..."
python3 main.py
