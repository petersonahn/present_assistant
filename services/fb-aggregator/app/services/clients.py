import os
import httpx

client: httpx.AsyncClient | None = None

# 게이트웨이 내부 네트워크 이름/포트에 맞춰 조정 가능
V1_FACE_URL = os.getenv("V1_URL", "http://v1-face:15010")
V2_EMOTION_URL = os.getenv("V2_URL", "http://v2-emotion:15011")
V3_POSE_URL = os.getenv("V3_URL", "http://v3-pose:15012")
A1_PROSODY_URL = os.getenv("A1_URL", "http://a1-prosody:15020")

async def init_client():
    global client
    if client is None:
        client = httpx.AsyncClient(timeout=5.0)

async def close_client():
    global client
    if client:
        await client.aclose()
        client = None

# 예: 필요시 내려 서비스 호출 유틸 (지금은 미사용)
async def call_health(url: str) -> dict:
    assert client
    try:
        r = await client.get(f"{url}/")
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}
