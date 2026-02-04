from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from mre_backend.api.routes import router
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.config import load_settings
from mre_backend.core.dataset_manager import DatasetManager
from mre_backend.core.run_manager import RunManager
from mre_backend.core.ws import WsManager

settings_bundle = load_settings()
settings = settings_bundle.settings

app = FastAPI(title="Model Reasoning Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

artifact_store = ArtifactStore(settings.paths.runs_dir)
ws_manager = WsManager()
run_manager = RunManager(settings, artifact_store, ws_manager)
dataset_manager = DatasetManager(settings, artifact_store)

app.state.settings = settings
app.state.artifact_store = artifact_store
app.state.run_manager = run_manager
app.state.dataset_manager = dataset_manager
app.state.ws_manager = ws_manager

app.include_router(router)


@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str):
    await ws_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        await ws_manager.disconnect(run_id, websocket)
