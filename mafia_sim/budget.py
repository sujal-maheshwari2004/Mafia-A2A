"""A shared per-game ceiling on LLM activity.

Without this, nothing stops a single game from making an unbounded number of
paid LLM calls -- a long discussion phase or a stuck loop could rack up cost
indefinitely. One `CallBudget` is shared across every `LLMAgent` at the
table; once it's exhausted, agents fall back to their existing cheap
defaults (random night action, abstain, silence) for the rest of the game.
"""

from __future__ import annotations

import time


class CallBudget:
    """A shared ceiling on LLM activity for one game.

    `record()` is called once per agent "turn" (night action / vote /
    discussion turn) that actually proceeds to call the LLM -- each such turn
    issues 1-3 real API calls (table read, day-summary, main call), so this is
    a proxy for total call volume, not an exact count. Once `exhausted`,
    agents fall back to their existing cheap defaults for the rest of the
    game.
    """

    def __init__(self, max_calls: int, max_seconds: float | None = None):
        self._max_calls = max_calls
        self._max_seconds = max_seconds
        self._calls = 0
        self._start = time.monotonic()

    @property
    def exhausted(self) -> bool:
        if self._calls >= self._max_calls:
            return True
        if self._max_seconds is not None and (time.monotonic() - self._start) > self._max_seconds:
            return True
        return False

    def record(self) -> None:
        self._calls += 1
