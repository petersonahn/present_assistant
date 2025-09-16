from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Set
import orjson

ws_router = APIRouter()

# 세션별 커넥션 관리
_connections: Dict[str, Set[WebSocket]] = {}

@ws_router.websocket("/ws")
async def ws_endpoint(ws: WebSocket, session_id: str = Query(...)):
    await ws.accept()
    _connections.setdefault(session_id, set()).add(ws)
    try:
        await ws.send_text(orjson.dumps({"ok": True, "msg": "fb-aggregator ws connected", "session_id": session_id}).decode())
        while True:
            # 필요 시 클라이언트에서 오는 메시지 처리
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _connections.get(session_id, set()).discard(ws)

async def broadcast_text(session_id: str, text: str):
    dead = []
    for c in list(_connections.get(session_id, set())):
        try:
            await c.send_text(text)
        except Exception:
            dead.append(c)
    for c in dead:
        _connections.get(session_id, set()).discard(c)

async def broadcast_json(session_id: str, payload: dict):
    text = orjson.dumps(payload).decode()
    await broadcast_text(session_id, text)
