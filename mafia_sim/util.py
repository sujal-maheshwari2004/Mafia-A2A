"""Small shared utilities."""

from __future__ import annotations

import hashlib


def stable_int_seed(*parts: object) -> int:
    """A reproducible integer derived from `parts`, stable across processes.

    Python's built-in `hash()` is salted per-process (for security), so
    `hash((name, seed))` produces a different value every run even with the
    same inputs -- making "reproducible" seeds not actually reproducible.
    This hashes a stable string representation instead.
    """
    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
