"""The three ways an agent can address the table."""

from __future__ import annotations

from enum import Enum


class CastType(Enum):
    UNICAST = "unicast"
    MULTICAST = "multicast"
    BROADCAST = "broadcast"

    @property
    def shorthand(self) -> str:
        return _SHORTHAND[self]


_SHORTHAND = {
    CastType.UNICAST: "uni",
    CastType.MULTICAST: "multi",
    CastType.BROADCAST: "broad",
}
