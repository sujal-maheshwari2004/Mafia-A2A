"""The data shapes that flow through the A2A protocol.

`CommRequest` is what an agent *hands the bus* when it decides to speak.
`Message` is the canonical, full-content record kept in the omniscient
spectator log. `Sighting` is what one agent actually *perceives* -- full
content if they were a party to it, or just its shape (who, how broad, to
whom) if they merely witnessed it pass between others.
"""

from __future__ import annotations

from dataclasses import dataclass

from .casting import CastType


@dataclass(frozen=True)
class CommRequest:
    """What an agent hands the bus when it decides to speak."""
    cast: CastType
    to: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class Message:
    """An entry in the omniscient spectator log -- full content, always."""
    seq: int
    sender: str
    cast: CastType
    to: tuple[str, ...]
    content: str
    day_number: int
    phase_label: str

    def shape(self) -> str:
        return f"{self.sender} -> {self.cast.shorthand}: {','.join(self.to)}"

    def __str__(self) -> str:
        return f"[{self.phase_label} {self.day_number}] {self.shape()} :: {self.content}"


@dataclass(frozen=True)
class Sighting:
    """An entry in one agent's personal feed.

    `content` is None when the agent merely witnessed the message pass
    between others -- they perceive its *shape* (who, how broad, to whom)
    but not what was said, just like overhearing a huddle across the table.
    """
    seq: int
    sender: str
    cast: CastType
    to: tuple[str, ...]
    content: str | None
    day_number: int
    phase_label: str

    @property
    def is_content_known(self) -> bool:
        return self.content is not None

    def render(self) -> str:
        targets = ",".join(self.to)
        header = f"[{self.phase_label} {self.day_number}] {self.sender} ({self.cast.shorthand} -> {targets})"
        if self.content is not None:
            return f"{header}: {self.content}"
        return f"{header} -- (you only see that this happened, not what was said)"
