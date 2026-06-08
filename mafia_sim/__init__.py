"""Mafia simulator: a reusable game engine, an A2A comm protocol, agents, and a CLI."""

from .agents import Agent, AgentView, DeathRecord, HeuristicAgent, LLMAgent, PERSONAS
from .events import (
    DayResolved,
    GameEnded,
    GameEvent,
    GameStarted,
    NightResolved,
    PhaseStarted,
    TableTalk,
    VoteCast,
)
from .protocol import CastType, CommBus, CommRequest, Message, ProtocolError, Sighting
from .engine import GameEngine, IllegalActionError
from .models import DayResult, Faction, NightResult, Phase, Player, Role
from .simulation import Simulation

__all__ = [
    "Agent",
    "AgentView",
    "DeathRecord",
    "HeuristicAgent",
    "LLMAgent",
    "PERSONAS",
    "DayResolved",
    "GameEnded",
    "GameEvent",
    "GameStarted",
    "NightResolved",
    "PhaseStarted",
    "TableTalk",
    "VoteCast",
    "CastType",
    "CommBus",
    "CommRequest",
    "Message",
    "ProtocolError",
    "Sighting",
    "GameEngine",
    "IllegalActionError",
    "DayResult",
    "Faction",
    "NightResult",
    "Phase",
    "Player",
    "Role",
    "Simulation",
]
