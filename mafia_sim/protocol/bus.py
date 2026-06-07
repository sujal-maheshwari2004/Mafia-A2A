"""The router at the heart of the A2A protocol.

`CommBus` is intentionally the *only* place that knows how to turn a raw
`CommRequest` into delivered `Sighting`s. It is completely agent-agnostic --
it has no notion of roles, factions, or game rules, only of who is currently
"present" (able to send and perceive at all) and who a message is addressed
to. That is what lets any kind of agent (rule-based, LLM-backed, human, or
something not yet imagined) share the exact same channel.

Pacing is a single global sequence counter stamped at submission time. That's
the entire "queue": ordering only matters where two messages land in the same
inbox, and a monotonic sequence number resolves that for free while costing
nothing where recipient sets never overlap.
"""

from __future__ import annotations

import itertools

from .casting import CastType
from .messages import CommRequest, Message, Sighting


class ProtocolError(ValueError):
    """Raised when a requested message violates the A2A protocol's rules."""


class CommBus:
    """Routes A2A messages and builds both the spectator log and per-agent feeds."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self.spectator_log: list[Message] = []
        self._feeds: dict[str, list[Sighting]] = {}

    def feed_for(self, name: str) -> list[Sighting]:
        return self._feeds.setdefault(name, [])

    def send(
        self,
        sender: str,
        request: CommRequest,
        *,
        present: tuple[str, ...],
        day_number: int,
        phase_label: str,
    ) -> Message:
        """Validate, route, and log one message. Returns the canonical record."""
        if sender not in present:
            raise ProtocolError(f"{sender} is not present right now and cannot speak")

        recipients = self._resolve_recipients(sender, request, present)

        seq = next(self._counter)
        message = Message(seq, sender, request.cast, recipients, request.content, day_number, phase_label)
        self.spectator_log.append(message)

        party = {sender, *recipients}
        for name in present:
            self.feed_for(name).append(
                Sighting(
                    seq=seq,
                    sender=sender,
                    cast=request.cast,
                    to=recipients,
                    content=request.content if name in party else None,
                    day_number=day_number,
                    phase_label=phase_label,
                )
            )

        return message

    @staticmethod
    def _resolve_recipients(sender: str, request: CommRequest, present: tuple[str, ...]) -> tuple[str, ...]:
        if request.cast is CastType.BROADCAST:
            if request.to:
                raise ProtocolError("broadcast addresses the whole room -- it takes no explicit recipients")
            return tuple(p for p in present if p != sender)

        recipients = tuple(dict.fromkeys(request.to))  # de-dupe, keep order
        if not recipients:
            raise ProtocolError(f"{request.cast.value} requires at least one named recipient")
        if sender in recipients:
            raise ProtocolError(f"{sender} cannot address themselves")
        if request.cast is CastType.UNICAST and len(recipients) != 1:
            raise ProtocolError("unicast must name exactly one recipient")
        if request.cast is CastType.MULTICAST and len(recipients) < 2:
            raise ProtocolError("multicast must name at least two recipients (use unicast for one)")
        for name in recipients:
            if name not in present:
                raise ProtocolError(f"{name} is not present right now and cannot be addressed")
        return recipients
