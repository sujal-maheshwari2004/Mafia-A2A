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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mafia_sim import PERSONAS, Agent, CallBudget, LLMAgent, Simulation, stable_int_seed
from mafia_sim.events import GameEvent

from .hub import GameHub

logger = logging.getLogger("mafia.director")

# Boundaries are picked on the hour in IST (the table's home timezone),
# so the schedule reads as a clean run of x:00s.
_IST = timezone(timedelta(hours=5, minutes=30))

# The fixed table everyone tunes in to see -- no viewer ever configures this.
PLAYER_NAMES = [
    "Avery", "Bailey", "Casey", "Drew", "Ellis",
    "Frankie", "Harper", "Jordan", "Kai", "Logan",
]
NUM_MAFIA = 2
MODEL = "gpt-4.1-mini"

# Safety net against a pathological game racking up cost: once either limit
# is hit, every agent falls back to cheap defaults for the rest of the game
# (see CallBudget / LLMAgent).
MAX_LLM_CALLS_PER_GAME = 400
MAX_GAME_SECONDS = 1800  # 30 minutes

# Sentinel placed on the bridge queue once the worker thread has nothing left to send.
_DONE = object()


@dataclass(frozen=True)
class _RolesRevealed:
    """Bridge-only marker: the freshly-seated table's true seat -> role mapping.

    Not a `GameEvent` -- it never goes out over the WebSocket, where roles stay
    secret until the table itself reveals them. It exists purely to hand the
    hub what `/game/roles` needs, the moment the table is seated.
    """

    roles: dict[str, str]


def _next_hour_boundary(now: datetime | None = None) -> datetime:
    """Next X:00 mark in IST -- the table's regular curtain time."""
    now = (now or datetime.now(timezone.utc)).astimezone(_IST)
    boundary = now.replace(minute=0, second=0, microsecond=0)
    if boundary <= now:
        boundary += timedelta(hours=1)
    return boundary.astimezone(timezone.utc)


def _build_agents(seed: int) -> list[Agent]:
    """Seat the table -- each player gets the same brain, a private rng, and one
    of the fifteen `PERSONAS` dealt out at random so the room sounds like a
    different group of people from game to game (never repeats a seat -- ten
    players, fifteen personas, no two voices the same on a given night)."""
    personas = random.Random(seed).sample(PERSONAS, len(PLAYER_NAMES))
    budget = CallBudget(MAX_LLM_CALLS_PER_GAME, MAX_GAME_SECONDS)
    return [
        LLMAgent(
            name,
            model=MODEL,
            persona=persona,
            rng=random.Random(stable_int_seed(name, seed)),
            budget=budget,
        )
        for name, persona in zip(PLAYER_NAMES, personas)
    ]


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
        sim = Simulation(agents, num_mafia=NUM_MAFIA, rng_seed=seed)
        events = sim.run()
        first = next(events)        # game_started -- the engine has now seated and assigned roles
        relay(first)
        relay(_RolesRevealed(sim.roles))
        for event in events:
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
        if isinstance(item, _RolesRevealed):
            hub.set_roles(item.roles)
            continue
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
