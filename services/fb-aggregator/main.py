from fastapi import FastAPI, WebSocket
app = FastAPI(title="Feedback Aggregator")

@app.get("/"), @app.get("/api/health")
def root(): return {"status":"ok","service":"fb-aggregator"}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    await ws.send_text("fb-aggregator ws connected")
    await ws.close()
