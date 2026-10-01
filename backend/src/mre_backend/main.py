from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mre_backend.api.routes import router
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.config import load_settings
from mre_backend.core.dataset_manager import DatasetManager
from mre_backend.core.run_manager import RunManager
from mre_backend.core.utils import safe_component
from mre_backend.core.ws import WsManager

LOCAL_ORIGINS = [
    f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (5173, 4173, 8000)
]


def create_app(settings=None) -> FastAPI:
    settings = settings or load_settings().settings
    artifact_store = ArtifactStore(settings.paths.runs_dir)
    ws_manager = WsManager()
    run_manager = RunManager(settings, artifact_store, ws_manager)
    dataset_manager = DatasetManager(settings, artifact_store)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        run_manager.executor.shutdown(wait=True, cancel_futures=True)
        dataset_manager.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Model Reasoning Explorer", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=LOCAL_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.settings = settings
    app.state.artifact_store = artifact_store
    app.state.run_manager = run_manager
    app.state.dataset_manager = dataset_manager
    app.state.ws_manager = ws_manager
    app.include_router(router)

    @app.middleware("http")
    async def local_mutations(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if origin is not None and origin not in LOCAL_ORIGINS:
                return JSONResponse(
                    {"detail": "Only the local UI may submit browser requests"}, 403
                )
            length = request.headers.get("content-length")
            limit = (settings.limits.max_upload_mb + 8) * 1024 * 1024
            if length is not None and (not length.isdigit() or int(length) > limit):
                return JSONResponse({"detail": "Request body too large"}, 413)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_request(_request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.websocket("/ws/runs/{run_id}")
    async def ws_run(websocket: WebSocket, run_id: str):
        origin = websocket.headers.get("origin")
        try:
            safe_component(run_id)
            if origin is not None and origin not in LOCAL_ORIGINS:
                raise ValueError("Untrusted origin")
        except ValueError:
            await websocket.close(code=1008)
            return
        await ws_manager.connect(run_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(run_id, websocket)

    return app


app = create_app()
