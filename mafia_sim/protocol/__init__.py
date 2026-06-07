"""The custom A2A (agent-to-agent) communication protocol.

Models how humans actually talk at a table: whoever is speaking picks, in the
moment, whether to address the whole room (`broadcast`), pull a few people
into a side huddle (`multicast`), or lean over and whisper to one neighbour
(`unicast`). Capability mirrors *presence* -- only agents who are "in the
room" right now can send or perceive anything -- and is otherwise completely
agent-agnostic: any agent (rule-based, LLM-backed, human, or otherwise) that
can produce a `CommRequest` can use it.

    from mafia_sim.protocol import CastType, CommBus, CommRequest

This package is the sole owner of the routing rules; `bus.py` documents them.
"""

from .bus import CommBus, ProtocolError
from .casting import CastType
from .messages import CommRequest, Message, Sighting

__all__ = [
    "CastType",
    "CommBus",
    "CommRequest",
    "Message",
    "ProtocolError",
    "Sighting",
]
