# services/fb-aggregator/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.ws.manager import ws_router
from app.services.clients import init_client, close_client

def create_app() -> FastAPI:
    app = FastAPI(title="Feedback Aggregator", version="0.1.0")

    # CORS (개발 편의)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(api_router)
    app.include_router(ws_router)

    # httpx AsyncClient 라이프사이클
    @app.on_event("startup")
    async def _on_startup():
        await init_client()

    @app.on_event("shutdown")
    async def _on_shutdown():
        await close_client()

    return app

app = create_app()