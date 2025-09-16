# app/ws/manager.py
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

ws_router = APIRouter()

class WSManager:
    def __init__(self) -> None:
        # 세션ID -> 연결된 WebSocket 집합
        self._sessions: Dict[str, Set[WebSocket]] = {}
        # latest 캐시(백업 폴링용)
        self._latest: Dict[str, dict] = {}

    async def connect(self, sid: str, ws: WebSocket) -> None:
        await ws.accept()
        self._sessions.setdefault(sid, set()).add(ws)

    def disconnect(self, sid: str, ws: WebSocket) -> None:
        try:
            conns = self._sessions.get(sid)
            if conns and ws in conns:
                conns.remove(ws)
                if not conns:
                    self._sessions.pop(sid, None)
        except Exception:
            pass  # 방어적 제거

    async def send(self, sid: str, data: dict) -> None:
        # 최신값 캐시
        self._latest[sid] = data
        # 세션에 열린 소켓들로 브로드캐스트
        conns = list(self._sessions.get(sid, []))
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(sid, ws)

    def latest(self, sid: str) -> dict:
        return self._latest.get(sid, {})

manager = WSManager()

@ws_router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    """표준 경로: /ws?session_id=..."""
    sid = session_id
    await manager.connect(sid, websocket)
    try:
        # 클라이언트에서 보내는 텍스트는 소비만(필요하면 처리 로직 추가)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(sid, websocket)

@ws_router.websocket("/ws/")
async def ws_endpoint_trailing_slash(websocket: WebSocket, session_id: str = Query(...)):
    """슬래시가 붙어도 동작하게 보조 엔드포인트"""
    await ws_endpoint(websocket, session_id)
