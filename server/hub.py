"""In-process broadcast hub for the one shared, scheduled Mafia game.

Exactly one `Simulation` plays at a time, on a background loop
(`director.direct_games`). Every viewer just subscribes here: each gets a
snapshot to catch up on instantly -- the live game's events so far, or the
last completed game's full transcript if nothing is running -- followed by a
live feed of whatever the hub broadcasts next. That's what lets someone tune
in mid-game and land in the middle of the action, or review the last game
while waiting for the next one, without the backend ever running more than
one simulation at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Union

from mafia_sim.events import GameEvent

Mode = Literal["idle", "live", "replay"]


@dataclass(frozen=True)
class StatusUpdate:
    """Broadcast on subscribe and whenever the hub's mode or schedule changes."""

    mode: Mode
    next_game_at: datetime | None


@dataclass(frozen=True)
class ErrorUpdate:
    """Broadcast when the live game hits trouble mid-stream."""

    message: str


@dataclass(frozen=True)
class HubSnapshot:
    """What a brand new subscriber needs to be caught up instantly."""

    events: list[GameEvent]
    mode: Mode
    next_game_at: datetime | None


BroadcastItem = Union[GameEvent, StatusUpdate, ErrorUpdate]


class GameHub:
    """Owns the current event log and fans it out to every subscriber.

    Every method here is plain, synchronous code that only ever runs on the
    asyncio event loop thread (the worker thread that actually plays a game
    only ever reaches it through `loop.call_soon_threadsafe`, same as the
    original per-connection bridge did). That means `subscribe` and
    `publish` can never interleave -- a new subscriber's snapshot is exactly
    "everything published before this call", and its queue is guaranteed to
    receive everything published after, with nothing skipped or duplicated.
    """

    def __init__(self) -> None:
        self._events: list[GameEvent] = []
        self._mode: Mode = "idle"
        self._next_game_at: datetime | None = None
        self._subscribers: set[asyncio.Queue[BroadcastItem]] = set()
        self._roles: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Subscriber lifecycle
    # ------------------------------------------------------------------
    def subscribe(self) -> tuple[HubSnapshot, "asyncio.Queue[BroadcastItem]"]:
        queue: asyncio.Queue[BroadcastItem] = asyncio.Queue()
        snapshot = HubSnapshot(events=list(self._events), mode=self._mode, next_game_at=self._next_game_at)
        self._subscribers.add(queue)
        return snapshot, queue

    def unsubscribe(self, queue: "asyncio.Queue[BroadcastItem]") -> None:
        self._subscribers.discard(queue)

    @property
    def roles(self) -> dict[str, str] | None:
        """Seat -> true role for the table currently (or most recently) seated.

        Backs the `/game/roles` reveal endpoint -- a deliberate, out-of-band
        peek behind the curtain for spectators who want one (a "reveal" toggle,
        a who's-who overlay, ...), independent of and well ahead of whatever the
        table itself has narratively figured out. `None` until the very first
        table is seated.
        """
        return dict(self._roles) if self._roles is not None else None

    def _broadcast(self, item: BroadcastItem) -> None:
        for queue in self._subscribers:
            queue.put_nowait(item)

    def _broadcast_status(self) -> None:
        self._broadcast(StatusUpdate(mode=self._mode, next_game_at=self._next_game_at))

    # ------------------------------------------------------------------
    # Director-driven transitions -- called only from `direct_games`
    # ------------------------------------------------------------------
    def schedule(self, next_game_at: datetime) -> None:
        """Announce when the next game starts without changing what's on display."""
        self._next_game_at = next_game_at
        self._broadcast_status()

    def start_game(self) -> None:
        self._events = []
        self._mode = "live"
        self._next_game_at = None
        self._broadcast_status()

    def set_roles(self, roles: dict[str, str]) -> None:
        """Record the freshly-seated table's seat -> role mapping for the reveal endpoint."""
        self._roles = dict(roles)

    def publish(self, event: GameEvent) -> None:
        self._events.append(event)
        self._broadcast(event)

    def report_error(self, message: str) -> None:
        self._broadcast(ErrorUpdate(message))

    def end_game(self, next_game_at: datetime) -> None:
        """The just-finished game's events stay put as the new "last game" --
        whoever connects before the next one starts gets to replay them."""
        self._mode = "replay"
        self._next_game_at = next_game_at
        self._broadcast_status()
