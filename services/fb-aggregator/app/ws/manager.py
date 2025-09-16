from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

ws_router = APIRouter()
_connections: Set[WebSocket] = set()

@ws_router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _connections.add(ws)
    try:
        await ws.send_text("fb-aggregator ws connected")
        while True:
            _ = await ws.receive_text()  # 필요시 클라이언트 메시지 처리
            # echo 또는 heartbeat 처리 등
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(ws)

async def broadcast(text: str):
    dead = []
    for c in _connections:
        try:
            await c.send_text(text)
        except Exception:
            dead.append(c)
    for c in dead:
        _connections.discard(c)
