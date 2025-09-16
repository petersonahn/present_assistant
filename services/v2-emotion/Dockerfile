FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx orjson pydantic==2.9.2
EXPOSE 15011
CMD ["bash", "-lc", "echo 'emotion container running'; tail -f /dev/null"]