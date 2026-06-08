"""FastAPI app that streams the one shared, scheduled Mafia game to everyone.

A fresh table of seven LLM-backed agents sits down every hour, on the hour,
played out on a single background loop (`director.direct_games`) -- there is
never more than one `Simulation` running. Every WebSocket connection simply
subscribes to the shared `GameHub`: it's handed a snapshot to catch up on
instantly (the live game's events so far, or the last completed game's full
transcript if the table's empty right now) and then a live feed of whatever
the hub broadcasts next, frame by frame, as the same JSON-serializable
`GameEvent`s `spectate.py` renders to a console. Nobody has to wait for a
fresh game to start to see what's going on at the table.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .director import direct_games
from .hub import ErrorUpdate, GameHub, StatusUpdate

logger = logging.getLogger("mafia.server")

hub = GameHub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set -- the scheduled LLM games will fail to play")

    task = asyncio.create_task(direct_games(hub))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Mafia Simulator",
    description="Streams the one shared, hourly agent-vs-agent Mafia game over a WebSocket -- live, or caught up.",
    lifespan=lifespan,
)


class _StatusFrame(BaseModel):
    type: Literal["status"] = "status"
    mode: Literal["idle", "live", "replay"]
    next_game_at: str | None = None


def _status_frame(update: StatusUpdate) -> dict:
    return _StatusFrame(
        mode=update.mode,
        next_game_at=update.next_game_at.isoformat() if update.next_game_at else None,
    ).model_dump(mode="json")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "info": "Open a WebSocket on /ws/game to watch the shared Mafia game -- live, or caught up on the last one.",
    }


@app.websocket("/ws/game")
async def stream_game(websocket: WebSocket) -> None:
    """Tune in to the one shared game, frame by frame, the moment you connect.

    No configuration message -- there's nothing to configure. The server
    starts streaming immediately. The very first frame is always

        {"type": "status", "mode": "idle" | "live" | "replay", "next_game_at": "<ISO8601>" | null}

    - `"live"`   -- a game is in progress. You're instantly caught up on
                    everything that's happened so far, then watch the rest
                    play out live.
    - `"replay"` -- the table's empty right now. What follows is the last
                    completed game's full transcript, then live coverage of
                    the next one once `next_game_at` arrives.
    - `"idle"`   -- the very first game hasn't started yet. Just wait for
                    `next_game_at`.

    After that, every frame is the `model_dump(mode="json")` of one
    `GameEvent` from `mafia_sim.events` (`game_started`, `phase_started`,
    `table_talk`, `night_resolved`, `day_resolved`, `game_ended`), in the
    exact order the game produced them -- interleaved with further `status`
    frames whenever the schedule changes, and `{"type": "error", "message":
    "..."}` if a game hits trouble mid-stream. The connection stays open
    across games, so you can simply leave it running; disconnect whenever
    you like.
    """
    await websocket.accept()
    snapshot, queue = hub.subscribe()

    try:
        await websocket.send_json(
            _status_frame(StatusUpdate(mode=snapshot.mode, next_game_at=snapshot.next_game_at))
        )
        for event in snapshot.events:
            await websocket.send_json(event.model_dump(mode="json"))

        while True:
            item = await queue.get()
            if isinstance(item, StatusUpdate):
                await websocket.send_json(_status_frame(item))
            elif isinstance(item, ErrorUpdate):
                await websocket.send_json({"type": "error", "message": item.message})
            else:
                await websocket.send_json(item.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue)
        with suppress(RuntimeError):
            await websocket.close()
