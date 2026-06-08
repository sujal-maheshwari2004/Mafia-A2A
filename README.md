# Mafia Simulator

A Mafia (a.k.a. Werewolf) engine, a custom agent-to-agent communication
protocol, pluggable AI players, and a live streaming server -- built up in
layers so each piece can be used on its own or wired into the next.

```
mafia_sim/
├── models.py       # Faction, Role, Player, Phase, NightResult, DayResult
├── engine.py       # GameEngine -- pure rules, no I/O
├── protocol/       # the A2A comm protocol (see below)
├── agents/         # the pluggable Agent interface + heuristic and LLM agents
├── events.py       # GameEvent types -- the streaming wire format
├── simulation.py   # Simulation -- plays a full agent-vs-agent game, yields events
└── cli.py          # human-playable text driver

server/             # FastAPI app that streams a live game over a WebSocket
main.py             # play one game yourself on the CLI
spectate.py         # watch agents play each other, narrated to the console
run_server.py       # serve games to a frontend over WebSocket
```

## Quick start

```bash
pip install fastapi uvicorn[standard] langchain-openai pydantic
```

**Play a game yourself** (you take one seat, bots fill the rest):

```bash
python main.py
```

**Watch AI agents play each other**, narrated to your terminal:

```bash
python spectate.py                       # 7 rule-based agents, random seed
python spectate.py 8 42                  # 8 players, fixed seed (reproducible)
python spectate.py 6 --brain llm         # LLM-backed agents (needs OPENAI_API_KEY)
python spectate.py 8 --brain mixed       # half LLM, half heuristic -- compare them directly
```

**Stream a game to a frontend:**

```bash
python run_server.py                     # serves on ws://127.0.0.1:8000/ws/game
```

## The game

Classic five-role Mafia: **Mafia**, **Doctor**, **Detective**, **Villager**.
Night phase (Mafia secretly choose a kill, Doctor protects, Detective
investigates) alternates with Day phase (open discussion, then a lynch vote),
until either the Town eliminates every Mafia or the Mafia equal or outnumber
the Town.

`engine.py` is pure game logic -- no `print`/`input` anywhere. It exposes
`start()`, `submit_night_action`/`resolve_night()`,
`submit_vote`/`resolve_day()`, win-condition checks, and a narrative `log`,
so a CLI, a test suite, or a server can all drive the exact same rules.

## The A2A protocol (`mafia_sim/protocol/`)

Agents don't just pick targets and votes -- they *talk*, the way people
actually do at a table, through a custom **agent-to-agent (A2A) protocol**
that is completely agent-agnostic (it knows nothing about roles, factions, or
who -- human or AI -- is behind a seat):

- **Three cast types**, freely chosen each time a player speaks:
  - `broadcast` -- address the whole room
  - `multicast` -- huddle with two or more specific people
  - `unicast` -- privately whisper to exactly one person
- **Capability mirrors presence**: at night, only the Mafia are "awake" and
  able to send or perceive anything at all; during the day, everyone still
  at the table can freely broadcast, huddle, or whisper.
- **Dual-layer visibility**: the omniscient *spectator log* (`CommBus.spectator_log`)
  records every message in full. Each agent's personal *feed*
  (`CommBus.feed_for(name)`) instead holds `Sighting`s -- full content if they
  were sender or recipient, but only the message's *shape* (who, what kind of
  cast, to whom) if they merely witnessed it pass between others. Choosing a
  private channel is therefore itself a visible, suspicion-fuelling move --
  exactly like being noticed leaning in to whisper at a real table.
- **Pacing** is a single global, monotonically increasing sequence number
  stamped at submission time -- the entire "queue". Ordering only matters
  where two messages land in the same inbox; everywhere else it costs nothing.

`CommBus.send(sender, request, *, present, day_number, phase_label)` is the
one place that turns a raw `CommRequest` into a routed `Message` (spectator
record) plus per-agent `Sighting`s, validating presence and recipients and
raising `ProtocolError` on any violation.

## Agents (`mafia_sim/agents/`)

Every seat is filled by something implementing the `Agent` ABC
(`choose_night_target`, `choose_vote`, `discussion_turn`), reasoning solely
from a read-only `AgentView` snapshot -- its own role/faction, who's alive,
what it has privately learned, and its personal A2A feed. Neither the
protocol nor the orchestrator (`Simulation`) ever know which kind of agent is
behind a seat, so they're fully interchangeable:

- **`HeuristicAgent`** -- rule-based, no API key required. Re-scans its feed
  live each turn for a keyword-driven "suspicion tally," coordinates Mafia
  kills by reading its teammates' night chatter, and produces templated
  accusations/whispers/claims across all three cast types.
- **`LLMAgent`** -- backed by **LangChain + OpenAI** (`ChatOpenAI(...).with_structured_output(...)`).
  A system prompt explains its identity, role, and the A2A protocol; a
  per-turn prompt supplies the live situation and its personal feed
  (rendered via `Sighting.render()`). Decisions come back as validated
  Pydantic objects, then get matched/repaired against the actual valid
  candidates (`_match_name`, `_resolve_recipients`) before ever reaching the
  bus, so a hallucinated name or malformed cast can never produce an illegal
  message. Requires `OPENAI_API_KEY` in the environment.

Mix and match freely -- `spectate.py --brain mixed` seats half of each so you
can compare them in the same game.

## Streaming events (`mafia_sim/events.py` + `simulation.py`)

`Simulation.run()` is a generator that **yields `GameEvent`s** instead of
printing -- it performs no I/O of its own, so the exact same simulation can
drive a console spectator, a test, or a live WebSocket. Every event is a
JSON-serializable Pydantic model with a literal `type` tag:

| `type`            | carries |
|-------------------|---------|
| `game_started`    | the seated roster |
| `phase_started`   | `Night`/`Day`, day number, who's present |
| `table_talk`      | one A2A message -- seq, sender, cast, recipients, content |
| `night_resolved`  | who (if anyone) died, whether the doctor's save worked |
| `vote_cast`       | one lynch vote landing live -- voter, target, the running tally so far |
| `day_resolved`    | who (if anyone) was lynched and their revealed role, vote tally, tie |
| `game_ended`      | the winning faction and everyone's final role |

Events are deliberately **structured rather than narrated** -- a frontend
decides *how* to say "Avery was lynched, and they were the Detective"
(copy, language, animation), not the backend. `spectate.py`'s `render_event`
is one example consumer; a React UI is another, fed the identical stream.

## Live streaming server (`server/`)

`server/app.py` is a small FastAPI app that streams *the one shared game*
over a single WebSocket endpoint, `/ws/game`. There's nothing to configure
and nothing to wait for:

- A fresh table of seven LLM-backed agents sits down **every hour, on the
  hour** (`server/director.py`'s `direct_games` loop) -- there is never more
  than one `Simulation` running on the server at a time.
- Every connection just subscribes to that one game's broadcast
  (`server/hub.py`'s `GameHub`). Connect any time and you're caught up
  *instantly*: mid-game, you get everything that's happened so far followed
  by the rest live; between games, you get the last completed game's full
  transcript followed by live coverage of the next one the moment it starts.

Connect and frames start flowing immediately -- no config message. The first
frame is always a status frame:

```json
{"type": "status", "mode": "live" | "replay" | "idle", "next_game_at": "2024-01-01T15:00:00Z" | null}
```

- `"live"` -- a game is in progress; what follows catches you up on it, then
  continues live.
- `"replay"` -- the table's empty right now; what follows is the last
  completed game's full transcript, then live coverage of the next one once
  `next_game_at` arrives.
- `"idle"` -- the very first game hasn't started yet; just wait for
  `next_game_at`.

After that, every frame is the `model_dump(mode="json")` of one `GameEvent`
from `mafia_sim.events` (`game_started`, `phase_started`, `table_talk`,
`night_resolved`, `vote_cast`, `day_resolved`, `game_ended`), in the exact
order the game produced them -- interleaved with further `status` frames
whenever the schedule changes, and `{"type": "error", "message": "..."}` if a
game hits trouble mid-stream. The connection stays open across games, so a
viewer can simply leave it running; disconnect whenever you like.

Voting itself is streamed live, not just as a final headline: each `vote_cast`
frame carries one player's vote (or abstention) plus the running tally the
instant after it lands, in casting order -- everything a frontend needs to
animate the count actually building (bars climbing, a leader emerging, a late
swing) rather than only being able to show the result once `day_resolved`
arrives with the final tally.

### Who is who: `GET /game/roles`

The WebSocket only ever reveals a seat's role when the *table itself* learns
it -- a lynch unmasks the victim, `game_ended` unmasks everyone, exactly as a
real game would. `GET /game/roles` is the deliberate exception: a plain JSON
endpoint, `{"roles": {"Avery": "Mafia", ...}}`, that reveals the truth behind
every seat at the currently- (or most recently-) seated table the moment it
sits down -- a peek behind the curtain for a spectator UI that wants to offer
one (a "reveal seats" toggle, a who's-who legend, face-down cards a viewer can
choose to flip for themselves). Returns `{"roles": null}` before the very
first table has ever been seated.

Because LLM-backed agents make blocking network calls, the scheduled game is
played out on its own background thread; its events are bridged onto the
asyncio event loop via a queue (`loop.call_soon_threadsafe`) and handed to
the hub, which fans them out to every subscriber, so a slow LLM call can
never stall the server or any viewer's socket.

Run it with:

```bash
python run_server.py [--host 0.0.0.0] [--port 8000] [--reload]
```

and point a frontend at `ws://<host>:<port>/ws/game`.

## Configuration

The scheduled game is LLM-backed, so the server needs an OpenAI API key in
the environment to play it:

```bash
export OPENAI_API_KEY=sk-...        # bash
$env:OPENAI_API_KEY = "sk-..."      # PowerShell
```

Without it, the hourly games will fail to play (the server logs a warning at
startup and an `error` frame is broadcast to viewers when a scheduled game
can't run).

`main.py`, `spectate.py`, and `mafia_sim`'s `HeuristicAgent` are unaffected
by any of this -- they still let you play or watch one-off games offline,
with whichever brain you like, straight from the CLI.
