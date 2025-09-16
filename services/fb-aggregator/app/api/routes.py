from fastapi import APIRouter
from fastapi.responses import JSONResponse

api_router = APIRouter()

@api_router.get("/")
@api_router.get("/api/health")
async def health():
    return {"status": "ok", "service": "fb-aggregator"}

# 이후 확장: 각 서비스 결과를 모아 오버레이 문구 생성
@api_router.post("/api/aggregate")
async def aggregate(payload: dict):
    """
    payload 예시:
    {
      "face": {...},        # v1-face 결과
      "emotion": {...},     # v2-emotion 결과
      "pose": {...},        # v3-pose 결과
      "prosody": {...}      # a1-prosody 결과
    }
    """
    # TODO: 규칙/스코어링 로직 추가
    # 아래는 더미 예시
    messages = []

    # 예: prosody
    prosody = payload.get("prosody") or {}
    if prosody.get("speaking_rate", 0) > 170:
        messages.append("말 속도가 조금 빨라요")
    elif prosody.get("speaking_rate", 0) < 120:
        messages.append("조금 더 또렷하게, 천천히 말해보세요")

    # 예: facial expression
    emotion = payload.get("emotion") or {}
    if emotion.get("smile_prob", 0) > 0.6:
        messages.append("미소가 자연스러워요")

    # 예: pose
    pose = payload.get("pose") or {}
    if pose.get("shoulder_balance") == "unbalanced":
        messages.append("어깨를 수평으로 맞춰보세요")

    # 기본 메시지
    if not messages:
        messages.append("좋아요! 지금 템포 유지해 보세요")

    result = {
        "messages": messages,
        "score": 75,  # TODO: 스코어 계산 로직
    }
    return JSONResponse(result)
