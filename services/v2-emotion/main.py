from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI(title="v2-emotion")

@app.get("/")
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "v2-emotion"}

@app.post("/infer")
async def infer(session_id: str = Form(...), frame: UploadFile = File(...)):
    """
    더미 엔드포인트: 이미지 업로드만 받고 OK 반환
    나중에 실제 감정 분석 모델 연결
    """
    data = await frame.read()
    return {
        "ok": True,
        "session_id": session_id,
        "frame_size": len(data),
        "emotion": "neutral"  # placeholder
    }
