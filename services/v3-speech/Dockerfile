FROM python:3.11-slim

# 시스템 패키지 설치 (오디오 처리 관련)
RUN apt-get update && apt-get install -y \
    # 오디오 라이브러리
    libasound2-dev \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    # FFmpeg (음성 처리)
    ffmpeg \
    # 기타 필수 패키지
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    # 개발 도구
    build-essential \
    pkg-config \
    # 한국어 처리 (KoNLPy)
    default-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치를 위한 환경 변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치 (오디오 관련 패키지 우선)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    # 오디오 처리 패키지 먼저 설치
    pip install --no-cache-dir sounddevice soundfile pyaudio && \
    # 나머지 패키지 설치
    pip install --no-cache-dir -r requirements.txt

# Whisper 모델 다운로드 (선택사항 - 런타임에 다운로드도 가능)
RUN python -c "import whisper; whisper.load_model('small')"

# 애플리케이션 파일 복사
COPY . .

# 포트 노출 (v3-pose와 다른 포트 사용)
EXPOSE 15013

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:15013/health || exit 1

# 서버 실행
CMD ["python3", "main.py"]
