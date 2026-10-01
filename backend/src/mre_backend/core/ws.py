from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WsManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, run_id: str, websocket: WebSocket) -> None:
        self._loop = asyncio.get_running_loop()
        await websocket.accept()
        self._connections[run_id].add(websocket)

    async def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        sockets = self._connections.get(run_id)
        if sockets is not None:
            sockets.discard(websocket)
            if not sockets:
                self._connections.pop(run_id, None)

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """Schedule a worker's event on the loop that owns the WebSocket."""
        loop = self._loop
        if loop is None or not loop.is_running() or loop.is_closed():
            return  # CLI runs do not have WebSocket clients.
        try:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(self.broadcast(run_id, event)))
        except RuntimeError:
            return  # The server may shut down between the loop checks.

    async def broadcast(self, run_id: str, event: dict[str, Any]) -> None:
        message = json.dumps(event)
        for socket in list(self._connections.get(run_id, [])):
            try:
                await socket.send_text(message)
            except Exception:
                await self.disconnect(run_id, socket)
