# app/api/routes.py
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Body
from fastapi.responses import JSONResponse
import orjson

from app.ws.manager import manager  # ← broadcast_json 대신 manager 사용

# /api 접두어로 고정
api_router = APIRouter(prefix="/api")

# 인메모리 세션 상태
SESS: Dict[str, Dict[str, Any]] = {}    # 원시 누적
LATEST: Dict[str, Dict[str, Any]] = {}  # 스무딩/가중치 적용 후 노출 상태

# 고성능 JSON 응답 유틸
def oj(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(content=orjson.loads(orjson.dumps(data)), status_code=status)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp(x: Optional[float], lo=0, hi=100) -> Optional[float]:
    if x is None:
        return None
    return max(lo, min(hi, x))

def ema(prev: Optional[float], new: Optional[float], alpha=0.4) -> Optional[float]:
    if new is None:
        return prev
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new

def compute_scores(raw: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    voice = raw.get("voice", {}) or {}
    face  = raw.get("face", {}) or {}
    pose  = raw.get("pose", {}) or {}
    emo   = raw.get("emotion", {}) or {}

    # 도메인별 스코어(가중 평균 등, 필요시 조정)
    voice_s0 = clamp(voice.get("clarity"))
    face_s0  = clamp((face.get("eye_contact", 0) * 0.7 + face.get("smile", 0) * 0.3))
    pose_s0  = clamp((pose.get("posture", 0) * 0.7 + pose.get("gesture_stability", 0) * 0.3))
    emo_s0   = clamp((emo.get("valence", 0) * 0.6 + (100 - abs(emo.get("arousal", 50) - 50)) * 0.4))

    # 이전 공개 값(LATEST) 기준 EMA
    prev_voice = (prev or {}).get("voice")
    prev_face  = (prev or {}).get("face")
    prev_pose  = (prev or {}).get("pose")
    prev_emo   = (prev or {}).get("emotion")

    voice_s = clamp(ema(prev_voice, voice_s0))
    face_s  = clamp(ema(prev_face, face_s0))
    pose_s  = clamp(ema(prev_pose, pose_s0))
    emo_s   = clamp(ema(prev_emo, emo_s0))

    # 종합 점수 가중합(결측 도메인은 자동 제외)
    weights = {"voice": 0.35, "pose": 0.35, "face": 0.2, "emotion": 0.1}
    parts = [("voice", voice_s), ("pose", pose_s), ("face", face_s), ("emotion", emo_s)]
    overall_val = 0.0
    total_w = 0.0
    for name, val in parts:
        if val is not None:
            overall_val += weights[name] * val
            total_w += weights[name]
    overall = clamp(round(overall_val if total_w > 0 else 0.0, 1))

    # 간단 규칙 기반 팁
    tips = []
    if voice.get("pace") and voice["pace"] > 85:
        tips.append("말 속도가 빨라요. 문장 사이에 짧은 호흡을 주세요.")
    if face.get("eye_contact") is not None and face["eye_contact"] < 60:
        tips.append("시선이 자주 흔들려요. 카메라 중앙을 바라봐요.")
    if pose.get("posture") is not None and pose["posture"] < 70:
        tips.append("어깨를 펴고 상체를 세워보세요.")
    if emo.get("valence") is not None and emo["valence"] < 50:
        tips.append("표정을 조금 더 부드럽게 유지해보세요.")
    if not tips:
        tips.append("좋아요! 지금 템포 유지해 보세요.")

    return {
        "ts": now_iso(),
        "overall": overall,
        "voice": voice_s,
        "face": face_s,
        "pose": pose_s,
        "emotion": emo_s,
        "tips": tips[:5],
    }

@api_router.get("/health")
async def health():
    return {"status": "ok", "service": "fb-aggregator"}

@api_router.get("/feedback/latest")
async def feedback_latest(session_id: str = Query(...)):
    """프론트 백업 폴링 엔드포인트"""
    return oj(LATEST.get(session_id, {}))

@api_router.post("/aggregate")
async def aggregate(payload: Dict[str, Any] = Body(...)):
    """
    게이트웨이가 face/emotion/pose/voice의 부분 결과를 세션 단위로 수집/병합/스무딩 후
    최신 결과를 LATEST[sid]에 저장하고, 해당 세션 구독자에게 WS로 브로드캐스트합니다.
    """
    sid = payload.get("session_id")
    if not sid:
        return oj({"ok": False, "error": "session_id required"}, status=400)

    cur = SESS.setdefault(sid, {})
    # 부분 업데이트 병합
    for k in ("face", "emotion", "pose", "voice"):
        if k in payload and isinstance(payload[k], dict):
            cur[k] = {**cur.get(k, {}), **payload[k]}

    # 스코어 계산(EMA는 기존 LATEST 기반)
    LATEST[sid] = compute_scores(cur, LATEST.get(sid))

    # 실시간 푸시 (WS)
    await manager.send(sid, LATEST[sid])

    return oj({"ok": True, "latest": LATEST[sid]})

# (선택) 수동 푸시 테스트용 엔드포인트
@api_router.post("/feedback/push")
async def feedback_push(
    session_id: str = Query(...),
    payload: Dict[str, Any] = Body(...)
):
    """
    수동 브로드캐스트(테스트/디버그용)
    예:
    curl -X POST 'http://fb-aggregator:15030/api/feedback/push?session_id=sid-123' \
         -H 'Content-Type: application/json' \
         -d '{"overall":82,"voice":75,"face":68,"tips":["말속도 좋아요","자세 안정적이에요"]}'
    """
    await manager.send(session_id, payload)
    # latest 캐시에도 반영하고 싶다면 주석 해제:
    # LATEST[session_id] = payload
    return oj({"ok": True})
