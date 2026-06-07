"""Core data types shared by the engine and any frontend (CLI, web, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Faction(Enum):
    TOWN = "Town"
    MAFIA = "Mafia"


class Role(Enum):
    MAFIA = "Mafia"
    DOCTOR = "Doctor"
    DETECTIVE = "Detective"
    VILLAGER = "Villager"


ROLE_FACTION = {
    Role.MAFIA: Faction.MAFIA,
    Role.DOCTOR: Faction.TOWN,
    Role.DETECTIVE: Faction.TOWN,
    Role.VILLAGER: Faction.TOWN,
}

# Roles that act at night, and what kind of action they take.
NIGHT_ACTION_ROLES = {Role.MAFIA, Role.DOCTOR, Role.DETECTIVE}


class Phase(Enum):
    LOBBY = "Lobby"
    NIGHT = "Night"
    DAY = "Day"
    ENDED = "Ended"


@dataclass
class Player:
    name: str
    role: Role
    alive: bool = True

    @property
    def faction(self) -> Faction:
        return ROLE_FACTION[self.role]

    def __str__(self) -> str:
        status = "alive" if self.alive else "dead"
        return f"{self.name} ({self.role.value}, {status})"


@dataclass
class NightResult:
    day_number: int
    killed: str | None
    saved: bool
    investigations: dict[str, tuple[str, Faction]] = field(default_factory=dict)


@dataclass
class DayResult:
    day_number: int
    lynched: str | None
    vote_counts: dict[str, int] = field(default_factory=dict)
    tied: bool = False
