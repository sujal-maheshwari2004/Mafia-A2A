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
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

from pydantic import TypeAdapter

from mafia_sim.events import GameEvent

logger = logging.getLogger("mafia.hub")

Mode = Literal["idle", "live", "replay"]

_EVENT_ADAPTER = TypeAdapter(GameEvent)

# Where the last completed game is persisted, so a restart can come back up
# in "replay" mode instead of a blank "idle" until the next scheduled game.
_DEFAULT_SAVE_PATH = Path(__file__).resolve().parent.parent / "data" / "last_game.json"

# Cap on how far a subscriber can fall behind before it's dropped -- keeps a
# stuck/slow consumer from growing its queue (and the hub's memory) without
# bound. A fresh subscriber gets the backlog via `HubSnapshot.events`, not
# this queue, so this only bounds *new* events arriving while connected.
_QUEUE_MAXSIZE = 256


class _Disconnected:
    """Sentinel: this subscriber's queue overflowed and it has been dropped."""


DISCONNECT = _Disconnected()


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


BroadcastItem = Union[GameEvent, StatusUpdate, ErrorUpdate, _Disconnected]


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

    def __init__(self, save_path: Path | None = None) -> None:
        self._events: list[GameEvent] = []
        self._mode: Mode = "idle"
        self._next_game_at: datetime | None = None
        self._subscribers: set[asyncio.Queue[BroadcastItem]] = set()
        self._roles: dict[str, str] | None = None
        self._save_path = save_path or _DEFAULT_SAVE_PATH
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Subscriber lifecycle
    # ------------------------------------------------------------------
    def subscribe(self) -> tuple[HubSnapshot, "asyncio.Queue[BroadcastItem]"]:
        queue: asyncio.Queue[BroadcastItem] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
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
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                self._drop_subscriber(queue)

    def _drop_subscriber(self, queue: "asyncio.Queue[BroadcastItem]") -> None:
        """A subscriber that's fallen too far behind to catch up -- disconnect
        it instead of blocking the hub or growing its queue without bound."""
        self._subscribers.discard(queue)
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(DISCONNECT)

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
        self._save_to_disk()

    # ------------------------------------------------------------------
    # Persistence -- survive a restart with the last completed game intact
    # ------------------------------------------------------------------
    def _load_from_disk(self) -> None:
        try:
            raw = json.loads(self._save_path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        try:
            self._events = [_EVENT_ADAPTER.validate_python(e) for e in raw["events"]]
            self._roles = raw.get("roles")
        except Exception:
            logger.warning("failed to load saved game from %s; ignoring", self._save_path, exc_info=True)
            return
        self._mode = "replay"

    def _save_to_disk(self) -> None:
        if not self._events:
            return  # don't clobber a good "last game" file with an empty/errored run
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "roles": self._roles,
                "events": [event.model_dump(mode="json") for event in self._events],
            }
            self._save_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.warning("failed to save game to %s", self._save_path, exc_info=True)
