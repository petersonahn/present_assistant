from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="a1-prosody")

@app.get("/")
def root():
    return {"status":"ok","service":"a1-prosody"}

@app.post("/infer")
async def infer(session_id: str = Form(...), audio: UploadFile = File(...)):
    _ = await audio.read()  # 실제 분석 전까지는 읽고 OK만
    return {"ok": True, "session_id": session_id, "wpm": 160, "volume_rms": 0.42}
