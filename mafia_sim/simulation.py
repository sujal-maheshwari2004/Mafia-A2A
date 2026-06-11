"""Runs a full agent-vs-agent Mafia game over the A2A protocol.

Every seat is filled by an `Agent`; there is no human player here (that's
what `cli.py` is for). This is the spectator path: it drives the `GameEngine`
through its phases and, during both the mafia's night huddle and the day's
table talk, lets agents converse freely through the `CommBus` -- exactly the
channel mix a human-in-the-loop game will eventually share.

`run()` is a generator that *yields* `GameEvent`s instead of printing. That
is what makes the exact same simulation drive a console spectator
(`spectate.py`) and a FastAPI WebSocket pushed straight to a browser: this
module never performs I/O of its own, it only narrates the game as a sequence
of structured, JSON-serializable moments and lets the consumer decide what to
do with them.

Pacing follows the "queue = submission order" design: each discussion is a
series of polling passes over the present agents (in a freshly shuffled order
each pass, so no one is always asked first); anyone may speak or stay silent;
a pass where nobody speaks ends the conversation.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Iterator

from .agents import Agent, AgentView, DeathRecord
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
from .protocol import CommBus, Message, ProtocolError
from .engine import GameEngine
from .models import Faction, Phase, Role

logger = logging.getLogger("mafia.simulation")

MAX_DISCUSSION_PASSES = 4

# A seat that's already sent this many messages this phase sits out the rest
# of it -- caps any one voice from dominating a discussion regardless of how
# many passes it takes to wind down.
MAX_MESSAGES_PER_PHASE = 3


class Simulation:
    def __init__(self, agents: list[Agent], num_mafia: int | None = None, rng_seed: int | None = None):
        self._agents = {agent.name: agent for agent in agents}
        self._rng = random.Random(rng_seed)
        self.engine = GameEngine([a.name for a in agents], num_mafia=num_mafia, rng_seed=rng_seed)
        self.bus = CommBus()
        self._knowledge: dict[str, dict[str, Faction]] = {a.name: {} for a in agents}
        self._dead: list[DeathRecord] = []
        # day_numbers where the doctor's protection foiled the mafia's kill --
        # the table doesn't learn who was attacked, only that someone was.
        self._saved_nights: list[int] = []
        # day_number -> {voter: target_or_None}, filled in as votes are cast.
        self._vote_history: dict[int, dict[str, str | None]] = {}

    @property
    def roles(self) -> dict[str, str]:
        """Seat -> true role, readable the instant the table is seated.

        The engine assigns roles inside `start()`, the first thing `run()`
        does -- so this is meaningful from the moment the `game_started` event
        has been yielded onward, well before any role is *narratively* revealed
        (a lynch, or the final `game_ended`). It exists for callers -- like the
        server's reveal endpoint -- that want to peek behind the curtain on
        purpose, independent of what the table itself has figured out.
        """
        return {p.name: p.role.value for p in self.engine.players}

    def run(self) -> Iterator[GameEvent]:
        """Play one game start to finish, yielding a `GameEvent` for every observable moment."""
        self.engine.start()
        self._seed_mafia_knowledge()
        yield GameStarted(players=list(self._agents))

        while self.engine.phase is not Phase.ENDED:
            if self.engine.phase is Phase.NIGHT:
                yield from self._run_night()
            else:
                yield from self._run_day()

        assert self.engine.winner is not None
        yield GameEnded(
            winner=self.engine.winner.value,
            roles={p.name: p.role.value for p in self.engine.players},
        )

    # ------------------------------------------------------------------
    # Night: only the mafia are "awake" -- they're the only ones who can
    # use the bus at all right now, exactly mirroring who's in the room.
    # ------------------------------------------------------------------
    def _run_night(self) -> Iterator[GameEvent]:
        mafia_present = tuple(p.name for p in self.engine.players_with_role(Role.MAFIA))
        # `present` is left empty here -- naming the awake players would hand
        # spectators the Mafia roster before the table itself ever learns it.
        yield PhaseStarted(
            phase="Night", day_number=self.engine.day_number, present=[], present_count=len(mafia_present)
        )

        if len(mafia_present) > 1:
            yield from self._run_discussion(mafia_present, Phase.NIGHT.value)

        for player in self.engine.alive_players:
            if player.role in (Role.MAFIA, Role.DOCTOR, Role.DETECTIVE):
                # Mafia are awake together; the Doctor and Detective each act alone,
                # with no one else awake to perceive them.
                present = mafia_present if player.role is Role.MAFIA else (player.name,)
                view = self._build_view(player.name, present=present)
                target = self._agents[player.name].choose_night_target(view)
                self.engine.submit_night_action(player.name, target)

        result = self.engine.resolve_night()
        for detective_name, (target_name, faction) in result.investigations.items():
            self._knowledge[detective_name][target_name] = faction
        if result.killed:
            self._dead.append(DeathRecord(
                name=result.killed,
                day_number=result.day_number,
                cause="killed in the night",
                revealed_role=None,
            ))
        elif result.saved:
            self._saved_nights.append(result.day_number)

        yield NightResolved(day_number=result.day_number, killed=result.killed, saved=result.saved)

    # ------------------------------------------------------------------
    # Day: everyone is at the table -- full freedom to broadcast, huddle,
    # or whisper, then a vote once the conversation naturally winds down.
    # ------------------------------------------------------------------
    def _run_day(self) -> Iterator[GameEvent]:
        present = tuple(p.name for p in self.engine.alive_players)
        yield PhaseStarted(
            phase="Day", day_number=self.engine.day_number, present=list(present), present_count=len(present)
        )

        yield from self._run_discussion(present, Phase.DAY.value)
        yield from self._run_vote(present)

        result = self.engine.resolve_day()
        lynched_role = self.engine.get_player(result.lynched).role.value if result.lynched else None
        if result.lynched:
            self._dead.append(DeathRecord(
                name=result.lynched,
                day_number=result.day_number,
                cause="lynched by the town's vote",
                revealed_role=lynched_role,
            ))
        yield DayResolved(
            day_number=result.day_number,
            lynched=result.lynched,
            lynched_role=lynched_role,
            vote_counts=result.vote_counts,
            tied=result.tied,
        )

    # ------------------------------------------------------------------
    # The vote itself, cast one player at a time and narrated live so a
    # frontend can visualize the tally building -- bars climbing, a leader
    # emerging, a late swing -- rather than only the final headline result.
    # ------------------------------------------------------------------
    def _run_vote(self, present: tuple[str, ...]) -> Iterator[GameEvent]:
        order = list(present)
        self._rng.shuffle(order)
        tally: Counter[str] = Counter()
        day_votes = self._vote_history.setdefault(self.engine.day_number, {})
        for name in order:
            view = self._build_view(name, present=present)
            target = self._agents[name].choose_vote(view)
            self.engine.submit_vote(name, target)
            day_votes[name] = target
            if target is not None:
                tally[target] += 1
            yield VoteCast(
                day_number=self.engine.day_number,
                voter=name,
                target=target,
                tally_so_far=dict(tally),
            )

    # ------------------------------------------------------------------
    # The "queue" is just submission order: poll present agents in a
    # reshuffled order each pass; route anything they send immediately;
    # stop once a full pass produces total silence.
    # ------------------------------------------------------------------
    def _run_discussion(self, present: tuple[str, ...], phase_label: str) -> Iterator[GameEvent]:
        order = list(present)
        # A "stale" pass is one where fewer than 1-in-4 seats spoke.
        # Two consecutive stale passes signals the conversation has run out of
        # new things to say.  We only start counting after the first active pass
        # so that a quiet opening pass (agents observing after a fresh death or
        # reveal) doesn't incorrectly cut the conversation short.
        min_speakers = max(1, len(present) // 4)
        had_active_pass = False
        stale_passes = 0
        speaker_counts: Counter[str] = Counter()
        for _ in range(MAX_DISCUSSION_PASSES):
            self._rng.shuffle(order)
            speakers_this_pass = 0
            for name in order:
                if speaker_counts[name] >= MAX_MESSAGES_PER_PHASE:
                    continue
                view = self._build_view(name, present=present)
                request = self._agents[name].discussion_turn(view)
                if request is None:
                    continue
                try:
                    message = self.bus.send(
                        name,
                        request,
                        present=present,
                        day_number=self.engine.day_number,
                        phase_label=phase_label,
                    )
                except ProtocolError as exc:
                    logger.warning("dropped invalid message from %s: %s", name, exc)
                    continue
                speaker_counts[name] += 1
                speakers_this_pass += 1
                yield self._table_talk_event(message, room_size=len(present))
            if speakers_this_pass == 0:
                break
            if speakers_this_pass >= min_speakers:
                had_active_pass = True
                stale_passes = 0
            elif had_active_pass:
                stale_passes += 1
                if stale_passes >= 2:
                    break

    @staticmethod
    def _table_talk_event(message: Message, room_size: int) -> TableTalk:
        return TableTalk(
            seq=message.seq,
            sender=message.sender,
            cast=message.cast.value,
            to=list(message.to),
            content=message.content,
            day_number=message.day_number,
            phase=message.phase_label,
            room_size=room_size,
        )

    # ------------------------------------------------------------------
    def _build_view(self, name: str, present: tuple[str, ...]) -> AgentView:
        player = self.engine.get_player(name)
        teammates: tuple[str, ...] = ()
        if player.role is Role.MAFIA:
            teammates = tuple(
                p.name for p in self.engine.players_with_role(Role.MAFIA, alive_only=False) if p.name != name
            )
        return AgentView(
            self_name=name,
            role=player.role,
            faction=player.faction,
            day_number=self.engine.day_number,
            phase=self.engine.phase,
            alive=tuple(p.name for p in self.engine.alive_players),
            present=present,
            dead=tuple(self._dead),
            saved_nights=tuple(self._saved_nights),
            teammates=teammates,
            known_factions=dict(self._knowledge[name]),
            feed=tuple(self.bus.feed_for(name)),
            vote_history=dict(self._vote_history),
        )

    def _seed_mafia_knowledge(self) -> None:
        mafia = self.engine.players_with_role(Role.MAFIA, alive_only=False)
        for player in mafia:
            for teammate in mafia:
                if teammate.name != player.name:
                    self._knowledge[player.name][teammate.name] = Faction.MAFIA
