from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WsManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, run_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[run_id].add(websocket)

    async def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if run_id in self._connections:
                self._connections[run_id].discard(websocket)

    async def broadcast(self, run_id: str, event: dict[str, Any]) -> None:
        message = json.dumps(event)
        async with self._lock:
            sockets = list(self._connections.get(run_id, []))
        for socket in sockets:
            try:
                await socket.send_text(message)
            except Exception:
                await self.disconnect(run_id, socket)
