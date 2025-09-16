import os
from typing import Optional, Dict, Any, List

import orjson
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------------------
# 환경변수 (docker-compose의 서비스 이름/포트를 기본값으로)
# ----------------------------
V1_URL = os.getenv("V1_URL", "http://v1-face:15010")
V2_URL = os.getenv("V2_URL", "http://v2-emotion:15011")
V3_URL = os.getenv("V3_URL", "http://v3-pose:15012")
A1_URL = os.getenv("A1_URL", "http://a1-prosody:15020")
FB_URL = os.getenv("FB_URL", "http://fb-aggregator:15030")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8.0"))

# ----------------------------
# FastAPI 앱
# ----------------------------
app = FastAPI(title="InterviewBuddy Gateway", openapi_url="/api/openapi.json", docs_url="/api/docs")

# CORS (프론트엔드에서 직접 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 고성능 JSON 응답
def oj(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=orjson.loads(orjson.dumps(data)), status_code=status_code)

# HTTP 클라이언트 (커넥션 재사용)
client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup():
    global client
    client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

@app.on_event("shutdown")
async def _shutdown():
    global client
    if client:
        await client.aclose()

# ----------------------------
# 헬스 & 연결 확인
# ----------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/ping-all")
async def ping_all():
    """
    현재 각 서비스와 연결이 되는지 GET / 로 빠르게 확인.
    (지금 services의 기본 main.py는 GET / 만 구현되어 있어 이 엔드포인트로 체크합니다.)
    """
    assert client is not None
    targets = {
        "v1-face": f"{V1_URL}/",
        "v2-emotion": f"{V2_URL}/",
        "v3-pose": f"{V3_URL}/",
        "a1-prosody": f"{A1_URL}/",
        "fb-aggregator": f"{FB_URL}/",
    }
    results: Dict[str, Any] = {}
    for name, url in targets.items():
        try:
            r = await client.get(url)
            results[name] = {"ok": r.status_code < 400, "status": r.status_code, "body": r.json() if r.headers.get("content-type","").startswith("application/json") else r.text}
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return results

# ----------------------------
# 분석 엔드포인트 (파일 업로드 → 각 서비스 fan-out)
# 주의: 각 서비스에 /infer 구현이 추가되어야 실제로 동작합니다.
#       (지금은 기본 템플릿에 GET /만 있으므로, 우선 /api/ping-all로 연결 체크하세요.)
# ----------------------------

@app.post("/api/ingest/frame")
async def ingest_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(...),
    use_services: Optional[List[str]] = Form(default=None),
):
    """
    한 프레임을 얼굴/감정/포즈 서비스에 동시에 전달.
    - session_id: 세션 식별자
    - frame: 이미지(JPEG/PNG 등) 바이너리
    - use_services: ["face","emotion","pose"] 중 선택. None이면 전부 호출
    """
    assert client is not None
    services_to_call = use_services or ["face", "emotion", "pose"]
    # 업로드 파일 바디 읽기 (한 번만 읽고 재사용)
    frame_bytes = await frame.read()
    files = {"frame": (frame.filename or "frame.jpg", frame_bytes, frame.content_type or "image/jpeg")}
    params = {"session_id": session_id}

    results: Dict[str, Any] = {}
    async def _call(name: str, base: str):
        url = f"{base}/infer"
        try:
            r = await client.post(url, params=params, files=files)
            return {"ok": r.status_code < 400, "status": r.status_code, "body": _safe_json(r)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if "face" in services_to_call:
        results["face"] = await _call("face", V1_URL)
    if "emotion" in services_to_call:
        results["emotion"] = await _call("emotion", V2_URL)
    if "pose" in services_to_call:
        results["pose"] = await _call("pose", V3_URL)

    return results

@app.post("/api/ingest/audio")
async def ingest_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    1초 등 오디오 청크를 prosody 서비스로 전달.
    - session_id: 세션 식별자
    - audio: wav/webm 등 바이너리
    """
    assert client is not None
    audio_bytes = await audio.read()
    files = {"audio": (audio.filename or "chunk.wav", audio_bytes, audio.content_type or "audio/wav")}
    params = {"session_id": session_id}
    try:
        r = await client.post(f"{A1_URL}/infer", params=params, files=files)
        return {"ok": r.status_code < 400, "status": r.status_code, "body": _safe_json(r)}
    except Exception as e:
        return oj({"ok": False, "error": str(e)}, status_code=502)

# ----------------------------
# 유틸
# ----------------------------
def _safe_json(r: httpx.Response) -> Any:
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}
    return {"raw": r.text}