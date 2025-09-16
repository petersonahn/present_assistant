#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install fastapi uvicorn[standard] httpx orjson pydantic==2.9.2
echo "✔ Dev container bootstrap completed"
