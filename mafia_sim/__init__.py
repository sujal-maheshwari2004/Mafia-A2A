"""Mafia simulator: a reusable game engine, an A2A comm protocol, agents, and a CLI."""

from .agents import Agent, AgentView, HeuristicAgent, LLMAgent
from .events import (
    DayResolved,
    GameEnded,
    GameEvent,
    GameStarted,
    NightResolved,
    PhaseStarted,
    TableTalk,
)
from .protocol import CastType, CommBus, CommRequest, Message, ProtocolError, Sighting
from .engine import GameEngine, IllegalActionError
from .models import DayResult, Faction, NightResult, Phase, Player, Role
from .simulation import Simulation

__all__ = [
    "Agent",
    "AgentView",
    "HeuristicAgent",
    "LLMAgent",
    "DayResolved",
    "GameEnded",
    "GameEvent",
    "GameStarted",
    "NightResolved",
    "PhaseStarted",
    "TableTalk",
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
