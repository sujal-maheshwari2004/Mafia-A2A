"""Structured, JSON-serializable events describing a game as it unfolds.

`Simulation.run()` yields these one at a time -- the exact same shapes whether
they end up printed to a terminal (`spectate.py`), pushed down a FastAPI
WebSocket to a browser, or inspected in a test. Every event is a Pydantic
model carrying a literal `type` tag, so a frontend can switch on `event.type`
and any layer can serialize the whole stream with `.model_dump(mode="json")`.

Deliberately fully structured rather than free-text: a frontend should decide
*how* to narrate "Avery was lynched" (copy, language, animation, ...), not be
handed pre-baked English to display verbatim.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class GameStarted(BaseModel):
    type: Literal["game_started"] = "game_started"
    players: list[str]


class PhaseStarted(BaseModel):
    type: Literal["phase_started"] = "phase_started"
    phase: Literal["Night", "Day"]
    day_number: int
    # Who is "in the room" -- able to send/perceive right now. Populated for Day
    # (the alive roster is already public); left empty for Night so spectators
    # can't read the Mafia's identity straight off this field. `present_count`
    # is always populated, even when `present` is hidden.
    present: list[str]
    present_count: int


class TableTalk(BaseModel):
    """One A2A message, exactly as the spectator log records it (full content)."""
    type: Literal["table_talk"] = "table_talk"
    seq: int
    sender: str
    cast: Literal["unicast", "multicast", "broadcast"]
    to: list[str]
    content: str
    day_number: int
    phase: Literal["Night", "Day"]
    # How many people could perceive this message (the audience for `cast`,
    # not just `len(to)`) -- e.g. a 2-person Mafia night huddle vs. a
    # full-table Day broadcast, even though both might be `cast="broadcast"`.
    room_size: int


class NightResolved(BaseModel):
    type: Literal["night_resolved"] = "night_resolved"
    day_number: int
    killed: str | None
    saved: bool


class VoteCast(BaseModel):
    """One lynch vote landing live -- the running tally as it stands the instant after it.

    Emitted as each player votes, in casting order, *before* `day_resolved`'s
    final tally -- exactly what a frontend needs to animate the vote building
    in real time (bars climbing, a leader emerging, a late swing) rather than
    only being able to show the result once the dust has settled.
    """
    type: Literal["vote_cast"] = "vote_cast"
    day_number: int
    voter: str
    target: str | None       # None = abstained
    tally_so_far: dict[str, int]


class DayResolved(BaseModel):
    type: Literal["day_resolved"] = "day_resolved"
    day_number: int
    lynched: str | None
    lynched_role: str | None
    vote_counts: dict[str, int]
    tied: bool


class GameEnded(BaseModel):
    type: Literal["game_ended"] = "game_ended"
    winner: Literal["Town", "Mafia"]
    roles: dict[str, str]  # player name -> role name, revealed once the game is over


GameEvent = Union[GameStarted, PhaseStarted, TableTalk, NightResolved, VoteCast, DayResolved, GameEnded]

__all__ = [
    "GameStarted",
    "PhaseStarted",
    "TableTalk",
    "NightResolved",
    "VoteCast",
    "DayResolved",
    "GameEnded",
    "GameEvent",
]
