FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx orjson pydantic==2.9.2
EXPOSE 15020
CMD ["bash", "-lc", "echo 'prosody container running'; tail -f /dev/null"]