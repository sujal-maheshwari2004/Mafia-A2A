"""Runs exactly one Mafia game an hour, on the hour, broadcasting it live.

`direct_games` is the long-running loop started at app startup: it sleeps
until the next hour boundary, plays one game off the event loop (LLM calls
block), and feeds its `GameEvent`s straight to the shared `GameHub` as they
happen -- then loops. Whatever the hub had on display before simply becomes
the new "last game" the moment this one ends, ready to replay for the next
viewer who tunes in between rounds.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from datetime import datetime, timedelta, timezone

from mafia_sim import Agent, LLMAgent, Simulation
from mafia_sim.events import GameEvent

from .hub import GameHub

logger = logging.getLogger("mafia.director")

# The fixed table everyone tunes in to see -- no viewer ever configures this.
PLAYER_NAMES = ["Avery", "Bailey", "Casey", "Drew", "Ellis", "Frankie", "Harper"]
MODEL = "gpt-4o-mini"

# Sentinel placed on the bridge queue once the worker thread has nothing left to send.
_DONE = object()


def _next_hour_boundary(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def _build_agents(seed: int) -> list[Agent]:
    return [LLMAgent(name, model=MODEL, rng=random.Random(hash((name, seed)) & 0xFFFF)) for name in PLAYER_NAMES]


def _play_in_background(seed: int, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    """Plays one whole game on a worker thread, relaying each event onto `loop`.

    Off the event loop because LLM-backed agents make blocking network calls
    -- the same bridge `server.app` used to run per-connection, just feeding
    the shared hub instead of one client's socket.
    """

    def relay(item: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        agents = _build_agents(seed)
        sim = Simulation(agents, rng_seed=seed)
        for event in sim.run():
            relay(event)
    except Exception as exc:  # noqa: BLE001 -- surfaced to viewers, not swallowed
        relay(exc)
    finally:
        relay(_DONE)


async def _run_one_game(hub: GameHub) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    seed = random.randrange(1_000_000)
    threading.Thread(target=_play_in_background, args=(seed, loop, queue), daemon=True).start()

    while True:
        item = await queue.get()
        if item is _DONE:
            return
        if isinstance(item, Exception):
            logger.error("scheduled game ended in error: %s", item)
            hub.report_error(str(item))
            return
        event: GameEvent = item
        hub.publish(event)


async def direct_games(hub: GameHub) -> None:
    """The hourly heartbeat: announce, sleep, play, broadcast, repeat -- forever."""
    next_at = _next_hour_boundary()
    hub.schedule(next_at)

    while True:
        delay = (next_at - datetime.now(timezone.utc)).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)

        hub.start_game()
        try:
            await _run_one_game(hub)
        except Exception:  # noqa: BLE001 -- a bad game shouldn't kill the heartbeat
            logger.exception("game director hit an unexpected error mid-game")

        next_at = _next_hour_boundary()
        hub.end_game(next_at)
