import os
import asyncio
import datetime
from typing import Optional, Dict, Any, List, Tuple

import orjson
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------------------
# 환경변수 (docker-compose의 서비스 이름/포트 기본값)
# ----------------------------
V1_URL = os.getenv("V1_URL", "http://v1-face:15010")
V2_URL = os.getenv("V2_URL", "http://v2-emotion:15011")
V3_URL = os.getenv("V3_URL", "http://v3-pose:15012")
A1_URL = os.getenv("A1_URL", "http://a1-prosody:15020")
FB_URL = os.getenv("FB_URL", "http://fb-aggregator:15030")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8.0"))
MAX_CONN = int(os.getenv("HTTP_MAX_CONN", "50"))
MAX_KEEPALIVE = int(os.getenv("HTTP_MAX_KEEPALIVE", "20"))

# ----------------------------
# FastAPI 앱
# ----------------------------
app = FastAPI(title="InterviewBuddy Gateway",
              openapi_url="/api/openapi.json",
              docs_url="/api/docs")

# CORS (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    limits = httpx.Limits(max_connections=MAX_CONN, max_keepalive_connections=MAX_KEEPALIVE)
    client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, limits=limits)

@app.on_event("shutdown")
async def _shutdown():
    global client
    if client:
        await client.aclose()
        client = None

# ----------------------------
# 유틸
# ----------------------------
def _safe_json(r: httpx.Response) -> Any:
    ct = (r.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}
    return {"raw": r.text}

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

async def _post_fb(payload: Dict[str, Any]) -> None:
    """fb-aggregator에 비동기 전송(집계/스코어링/WS 푸시는 aggregator가 수행). 실패해도 게이트웨이 응답엔 영향 없음."""
    assert client is not None
    try:
        await client.post(f"{FB_URL}/api/aggregate", json=payload)
    except Exception:
        # 로깅 필요 시 여기에 추가
        pass

# ----------------------------
# 헬스 & 연결 확인
# ----------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/ping-all")
async def ping_all():
    """
    각 서비스의 /api/health 호출 결과 집계.
    """
    assert client is not None
    targets = {
        "v1-face": f"{V1_URL}/api/health",
        "v2-emotion": f"{V2_URL}/api/health",
        "v3-pose": f"{V3_URL}/api/health",
        "a1-prosody": f"{A1_URL}/api/health",
        "fb-aggregator": f"{FB_URL}/api/health",
    }

    async def _ping(name: str, url: str) -> Tuple[str, Dict[str, Any]]:
        try:
            r = await client.get(url)
            body = _safe_json(r)
            return name, {"ok": r.status_code < 400, "status": r.status_code, "body": body}
        except Exception as e:
            return name, {"ok": False, "error": str(e)}

    results: Dict[str, Any] = {}
    for coro in asyncio.as_completed([_ping(n, u) for n, u in targets.items()]):
        name, res = await coro
        results[name] = res
    return results

# ----------------------------
# 분석 엔드포인트 (파일 업로드 → 각 서비스 fan-out)
# ----------------------------

@app.post("/api/ingest/frame")
async def ingest_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(...),
    use_services: Optional[List[str]] = Form(default=None),
):
    """
    한 프레임을 얼굴/감정/포즈 서비스에 동시에 전달하여 결과를 수집.
    - session_id: 세션 식별자
    - frame: 이미지(JPEG/PNG 등) 바이너리
    - use_services: ["face","emotion","pose"] 중 선택. None이면 전부 호출
    """
    assert client is not None
    services_to_call = use_services or ["face", "emotion", "pose"]

    # 업로드 파일 바디 읽기 (한 번만 읽고 재사용)
    frame_bytes = await frame.read()
    files = {
        "frame": (
            frame.filename or "frame.jpg",
            frame_bytes,
            frame.content_type or "image/jpeg",
        )
    }
    params = {"session_id": session_id}

    async def _call(name: str, base: str) -> Tuple[str, Dict[str, Any]]:
        url = f"{base}/infer"
        try:
            r = await client.post(url, params=params, files=files)
            return name, {"ok": r.status_code < 400, "status": r.status_code, "body": _safe_json(r)}
        except Exception as e:
            return name, {"ok": False, "error": str(e)}

    tasks = []
    if "face" in services_to_call:    tasks.append(_call("face", V1_URL))
    if "emotion" in services_to_call: tasks.append(_call("emotion", V2_URL))
    if "pose" in services_to_call:    tasks.append(_call("pose", V3_URL))

    results: Dict[str, Any] = {}
    for coro in asyncio.as_completed(tasks):
        name, res = await coro
        results[name] = res

    # fb-aggregator에 비동기로 전달(부분 업데이트 허용)
    agg_payload: Dict[str, Any] = {"session_id": session_id, "ts": _now_iso()}
    if results.get("face", {}).get("ok"):    agg_payload["face"] = results["face"]["body"]
    if results.get("emotion", {}).get("ok"): agg_payload["emotion"] = results["emotion"]["body"]
    if results.get("pose", {}).get("ok"):    agg_payload["pose"] = results["pose"]["body"]
    if len(agg_payload) > 2:  # session_id, ts 외 내용이 있을 때만 전송
        asyncio.create_task(_post_fb(agg_payload))

    return results

@app.post("/api/ingest/audio")
async def ingest_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    1초 등 오디오 청크를 prosody 서비스로 전달하여 결과 수집.
    """
    assert client is not None
    audio_bytes = await audio.read()
    files = {
        "audio": (
            audio.filename or "chunk.wav",
            audio_bytes,
            audio.content_type or "audio/wav",
        )
    }
    params = {"session_id": session_id}

    try:
        r = await client.post(f"{A1_URL}/infer", params=params, files=files)
        body = _safe_json(r)
        ok = r.status_code < 400
        # 성공 시 aggregator에 voice 파트 전달
        if ok:
            asyncio.create_task(_post_fb({
                "session_id": session_id,
                "ts": _now_iso(),
                "voice": body
            }))
        return {"ok": ok, "status": r.status_code, "body": body}
    except Exception as e:
        return oj({"ok": False, "error": str(e)}, status_code=502)