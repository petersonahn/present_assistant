FROM python:3.11-slim
WORKDIR /app
# 앱 소스 복사 (최소 main.py는 꼭 있어야 함)
COPY main.py .
# 의존성 설치
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx orjson pydantic==2.9.2 python-multipart
EXPOSE 15000
# uvicorn으로 서버 실행
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","15000"]