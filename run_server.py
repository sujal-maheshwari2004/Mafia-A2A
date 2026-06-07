"""Serve the Mafia simulator over HTTP/WebSocket for a frontend to connect to.

Usage:
    python run_server.py [--host 0.0.0.0] [--port 8000] [--reload]

Connect a browser or React app to ws://<host>:<port>/ws/game -- see
`server.app.stream_game` for the wire protocol.
"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on source changes (development)")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
