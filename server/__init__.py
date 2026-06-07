"""FastAPI app that streams live Mafia games to a frontend over a WebSocket."""

from .app import app

__all__ = ["app"]
