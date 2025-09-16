FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# OpenCV 런타임 의존성 (headless 사용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 및 모델 파일 복사
COPY . .

EXPOSE 15012

# FastAPI(Uvicorn)로 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "15012"]