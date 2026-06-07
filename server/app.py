"""FastAPI app that streams a live Mafia game to any connected frontend.

A game is synchronous, and once LLM-backed agents are seated it makes
blocking network calls (LangChain's `.invoke()`), so each game is played out
on its own background thread. Its `GameEvent`s -- the same structured,
JSON-serializable moments `spectate.py` renders to a console -- are bridged
onto the asyncio event loop through a queue, and the WebSocket endpoint just
forwards each one to the browser as JSON, in the exact order the game
produced it. Neither the simulation nor the protocol know or care that a
WebSocket is on the other end; that's what the whole `GameEvent` refactor was
for.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from mafia_sim import Agent, HeuristicAgent, LLMAgent, Simulation
from mafia_sim.events import GameEvent

DEFAULT_NAMES = [
    "Avery", "Bailey", "Casey", "Drew", "Ellis",
    "Frankie", "Harper", "Jordan", "Kai", "Logan",
]

# Sentinel placed on the bridge queue once the worker thread has nothing left to send.
_DONE = object()

app = FastAPI(title="Mafia Simulator", description="Streams live agent-vs-agent Mafia games over a WebSocket.")


class GameConfig(BaseModel):
    """What the client sends right after opening the socket to configure its game."""
    count: int = 7
    seed: int | None = None
    brain: Literal["heuristic", "llm", "mixed"] = "heuristic"
    model: str = "gpt-4o-mini"


def _build_agent(brain: str, name: str, seed: int, model: str) -> Agent:
    rng = random.Random(hash((name, seed)) & 0xFFFF)
    if brain == "llm":
        return LLMAgent(name, model=model, rng=rng)
    if brain == "mixed":
        # alternate so a frontend can directly compare LLM vs. heuristic play in one game
        return LLMAgent(name, model=model, rng=rng) if hash(name) % 2 else HeuristicAgent(name, rng=rng)
    return HeuristicAgent(name, rng=rng)


def _play_in_background(config: GameConfig, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    """Plays one whole game on a worker thread, relaying each event onto `loop`.

    Runs entirely off the event loop so a slow LLM call can never stall the
    WebSocket or any other connection the server is handling.
    """

    def relay(item: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        seed = config.seed if config.seed is not None else random.randrange(1_000_000)
        names = DEFAULT_NAMES[: config.count]
        agents = [_build_agent(config.brain, name, seed, config.model) for name in names]
        sim = Simulation(agents, rng_seed=seed)
        for event in sim.run():
            relay(event)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client, not swallowed
        relay(exc)
    finally:
        relay(_DONE)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "info": "Open a WebSocket on /ws/game and send a config message to watch a live game."}


@app.websocket("/ws/game")
async def stream_game(websocket: WebSocket) -> None:
    """Configure and watch one game live, frame by frame.

    Protocol: connect, then send one JSON config message --
    `{"count": 7, "seed": 42, "brain": "heuristic", "model": "gpt-4o-mini"}`
    (every field optional, defaults shown). The server replies with a stream
    of JSON frames, each the `model_dump(mode="json")` of one `GameEvent`
    from `mafia_sim.events` (`game_started`, `phase_started`, `table_talk`,
    `night_resolved`, `day_resolved`, `game_ended` -- discriminated by their
    `type` field), in the exact order the game produced them, until the game
    ends and the socket closes. A malformed config or a game-side error is
    sent as `{"type": "error", "message": "..."}` before closing.
    """
    await websocket.accept()

    try:
        raw_config = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    try:
        config = GameConfig.model_validate(raw_config or {})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"invalid game config: {exc}"})
        await websocket.close()
        return

    if config.brain in ("llm", "mixed") and not os.environ.get("OPENAI_API_KEY"):
        await websocket.send_json({"type": "error", "message": "OPENAI_API_KEY is not set on the server"})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    threading.Thread(target=_play_in_background, args=(config, loop, queue), daemon=True).start()

    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                await websocket.send_json({"type": "error", "message": str(item)})
                break
            event: GameEvent = item
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed -- e.g. the client disconnected first
