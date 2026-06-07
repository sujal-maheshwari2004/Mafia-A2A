"""The pluggable agent interface -- the same shape for rule-based and LLM agents.

An `AgentView` is the *only* thing an agent ever sees: a read-only snapshot of
what it personally knows at the moment it must decide. It carries no engine
internals, so any agent implementation (heuristic today, LLM-backed tomorrow)
reasons solely from information a player at the table could plausibly have.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..protocol import CommRequest, Sighting
from ..models import Faction, Phase, Role


@dataclass(frozen=True)
class AgentView:
    self_name: str
    role: Role
    faction: Faction
    day_number: int
    phase: Phase
    alive: tuple[str, ...]
    teammates: tuple[str, ...]              # fellow mafia, if you are one
    known_factions: dict[str, Faction]      # learned via investigation, etc.
    feed: tuple[Sighting, ...]              # your personal A2A perception history

    @property
    def others_alive(self) -> tuple[str, ...]:
        return tuple(name for name in self.alive if name != self.self_name)


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_night_target(self, view: AgentView) -> str:
        """Name of the player to act on tonight (kill / protect / investigate)."""

    @abstractmethod
    def choose_vote(self, view: AgentView) -> str | None:
        """Name of the player to lynch today, or None to abstain."""

    @abstractmethod
    def discussion_turn(self, view: AgentView) -> CommRequest | None:
        """A message to send right now, or None to stay silent this turn."""
