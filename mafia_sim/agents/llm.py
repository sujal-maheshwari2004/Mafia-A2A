"""LLM-backed agent (LangChain + OpenAI).

Implements the exact same `Agent` interface as `HeuristicAgent` -- the comm
protocol and the simulation orchestrator are completely agent-agnostic, so
this is a drop-in swap (and any future agent, on any stack, plugs in the same
way). It reasons solely from its `AgentView`: a system prompt explains its
identity, the rules, and the A2A protocol; a per-turn human prompt supplies
the live situation -- who's alive, what it has personally perceived -- and
the question at hand. Decisions are returned as structured Pydantic objects
so they can be validated and mapped back onto the protocol's exact types.

Requires `OPENAI_API_KEY` in the environment (langchain-openai reads it
automatically).
"""

from __future__ import annotations

import random
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..protocol import CastType, CommRequest
from ..models import Role
from .base import Agent, AgentView

DEFAULT_MODEL = "gpt-4o-mini"

# How many of an agent's own past private notes get fed back into its next
# prompt -- enough continuity to remember a running suspicion without letting
# the prompt grow without bound over a long game.
_MEMORY_LIMIT = 14

_REASONING_FIELD = (
    "Your own private take on this moment -- who you suspect and why, what you're "
    "tracking, what you plan to do about it. Never shown to anyone else, but it IS "
    "kept as your own running notes for the rest of the game, so write it the way "
    "you'd actually want to be reminded of your own read later: specific, and honest "
    "about your suspicions, doubts, and hunches -- not a tidy summary for an audience."
)


class _NightActionDecision(BaseModel):
    target: str = Field(description="Exact name of the player you act on tonight")
    reasoning: str = Field(description=_REASONING_FIELD)


class _VoteDecision(BaseModel):
    target: str | None = Field(default=None, description="Exact name of the player to lynch, or null to abstain")
    reasoning: str = Field(description=_REASONING_FIELD)


class _DiscussionDecision(BaseModel):
    speak: bool = Field(description="True to say something now, False to stay silent and pass this turn")
    cast: Literal["unicast", "multicast", "broadcast"] = Field(
        default="broadcast",
        description="unicast = whisper to exactly one person, multicast = huddle with two or more, "
                    "broadcast = address the whole room",
    )
    to: list[str] = Field(
        default_factory=list,
        description="Recipients by exact name: exactly one for unicast, two or more for multicast, "
                    "empty for broadcast",
    )
    content: str = Field(default="", description="What you actually say, in natural conversational language")
    reasoning: str = Field(description=_REASONING_FIELD)


_ROLE_BRIEFS = {
    Role.MAFIA: (
        "You are MAFIA. Each night you and your teammates secretly choose one player to eliminate. "
        "You win once the Mafia equal or outnumber the Town. Blend in, deflect suspicion, and "
        "steer the Town's votes toward townsfolk -- especially anyone who seems to be figuring you out."
    ),
    Role.DOCTOR: (
        "You are the DOCTOR. Each night you may protect one player (including yourself) from the "
        "Mafia's attack. You win when the Town eliminates all Mafia. You usually won't know who the "
        "Mafia are targeting, so use your nights wisely."
    ),
    Role.DETECTIVE: (
        "You are the DETECTIVE. Each night you may investigate one player to learn whether they are "
        "Town or Mafia. You win when the Town eliminates all Mafia. Deciding when -- and how -- to "
        "share what you learn is the core of the role: claim too early and the Mafia will have you "
        "lynched; wait too long and the Town may run out of time."
    ),
    Role.VILLAGER: (
        "You are a VILLAGER with no special power -- only your voice, your read on people, and your "
        "vote. You win when the Town eliminates all Mafia. Listen closely, build trust, and vote out "
        "whoever you believe is Mafia."
    ),
}

_PROTOCOL_BRIEF = """
COMMUNICATION -- you speak the way a real person at the table would, choosing freely each time:
  - broadcast : address everyone present in the room right now
  - multicast : pull two or more specific people into a side huddle
  - unicast   : lean over and privately whisper to exactly one person

Anyone you do NOT address can still tell that you sent a private message and who
it went to (its "shape"), even though they can never see what you actually said.
Choosing a private channel is therefore itself a visible move that others may read
into -- exactly like being noticed whispering at a real table.

Capability mirrors presence: at night only the Mafia are awake and able to talk at
all (everyone else is asleep and perceives nothing); during the day everyone still
at the table can freely broadcast, huddle, or whisper.

SCHEME -- don't just stand in the middle of the room saying everything to everyone.
That is not how people who need allies, cover, or an alibi actually operate:
  - Pull someone aside BEFORE you commit to a position out loud, to test a read,
    trade notes, or get a feel for whether they'd back you if you spoke up.
  - Build a quiet alliance with one or two people you've started to trust, and
    actually lean on it later -- back each other's votes, cross-confirm a story,
    warn each other privately when the room turns.
  - Drop a seed of doubt about someone in a whisper or a huddle and watch whether
    it resurfaces in the open later -- that tells you who talks, and to whom.
  - If you're MAFIA, the scheming never stops at sunrise: the day is yours too --
    coordinate your cover story, decide together (quietly) who to feed to the
    town's suspicion, and steer votes through side-channels as much as the room.
A table where every single message is a broadcast is not a real one -- vary your
channel on purpose, and let WHO you choose to involve be part of your strategy.
""".strip()

_VOICE_BRIEF = """
HOW YOU SOUND -- you are a person sitting at this table with your own neck on the
line, not an assistant describing one from the outside. Talk like it:
  - A death should land like a gut-punch, not a data point: "wait -- they got
    CASEY? Last night? No, no, that doesn't... who would even--" beats "Casey's
    death raises some interesting questions."
  - A reveal should visibly rearrange your head on the spot: "hold on. HOLD ON.
    Harper was the Doctor? Then who has been protecting--" beats a calm recap.
  - Let yourself be perplexed when something doesn't add up, rattled when the
    finger swings toward you, intrigued when a thread you've been pulling on
    finally clicks. Push back, get defensive, needle the people you don't trust,
    second-guess yourself out loud, trail off mid-thought when you realize
    something live.
  - Skip the even, hedged, neatly-bulleted register of an assistant summarizing
    a situation for someone else -- nobody whose life is on the line sounds like
    that. Sound like you're actually IN it.
""".strip()


class LLMAgent(Agent):
    """An agent whose decisions are produced by an LLM via LangChain + OpenAI."""

    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.85,
        rng: random.Random | None = None,
    ):
        super().__init__(name)
        self._rng = rng or random.Random()
        # Continuity of suspicion: every private "reasoning" the model writes
        # gets kept here and re-served back to it next turn, so a read formed
        # on Day 1 can still shape a vote on Day 3 instead of being re-derived
        # -- or quietly forgotten -- from scratch each time.
        self._memory: list[str] = []
        llm = ChatOpenAI(model=model, temperature=temperature)
        self._night_brain = llm.with_structured_output(_NightActionDecision)
        self._vote_brain = llm.with_structured_output(_VoteDecision)
        self._discussion_brain = llm.with_structured_output(_DiscussionDecision)

    # ------------------------------------------------------------------
    # Night
    # ------------------------------------------------------------------
    def choose_night_target(self, view: AgentView) -> str:
        candidates = self._night_candidates(view)
        question = {
            Role.MAFIA: "Night falls. Who do you and your fellow Mafia choose to eliminate tonight? "
                        "(You cannot target your own teammates.)",
            Role.DOCTOR: "Night falls. Who do you protect tonight? (You may choose yourself.)",
            Role.DETECTIVE: "Night falls. Who do you investigate tonight?",
        }[view.role]

        try:
            decision = self._night_brain.invoke(self._messages(view, question))
        except Exception as exc:
            self._warn(f"night-action call failed ({exc!r}); choosing at random")
            return self._rng.choice(candidates)

        self._remember(view, decision.reasoning)
        return self._match_name(decision.target, candidates) or self._rng.choice(candidates)

    def _night_candidates(self, view: AgentView) -> list[str]:
        others = list(view.others_alive)
        if view.role is Role.MAFIA:
            return [n for n in others if n not in view.teammates] or others
        if view.role is Role.DOCTOR:
            return [view.self_name, *others]
        return others  # Detective

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------
    def choose_vote(self, view: AgentView) -> str | None:
        others = list(view.others_alive)
        if not others:
            return None

        question = (
            "It's time to vote. Who do you vote to lynch today? If you genuinely have no read, "
            "you may abstain by leaving the target null."
        )
        try:
            decision = self._vote_brain.invoke(self._messages(view, question))
        except Exception as exc:
            self._warn(f"vote call failed ({exc!r}); abstaining")
            return None

        self._remember(view, decision.reasoning)
        if decision.target is None:
            return None
        return self._match_name(decision.target, others)

    # ------------------------------------------------------------------
    # Discussion
    # ------------------------------------------------------------------
    def discussion_turn(self, view: AgentView) -> CommRequest | None:
        others = list(view.others_alive)
        if not others:
            return None

        question = (
            "It's your moment to optionally speak. Decide whether to say something right now, and "
            "if so, choose your channel and audience deliberately -- broadcasting, huddling, and "
            "whispering each send a different signal to the room. If you'd rather wait and listen, pass."
        )
        try:
            decision = self._discussion_brain.invoke(self._messages(view, question))
        except Exception as exc:
            self._warn(f"discussion call failed ({exc!r}); staying silent")
            return None

        self._remember(view, decision.reasoning)
        if not decision.speak or not decision.content.strip():
            return None

        try:
            cast = CastType(decision.cast)
        except ValueError:
            cast = CastType.BROADCAST
        if cast is CastType.MULTICAST and len(others) < 2:
            cast = CastType.UNICAST

        to = self._resolve_recipients(cast, decision.to, others)
        return CommRequest(cast, to, decision.content.strip())

    def _resolve_recipients(self, cast: CastType, raw_to: list[str], others: list[str]) -> tuple[str, ...]:
        if cast is CastType.BROADCAST:
            return ()

        matched: list[str] = []
        for raw in raw_to:
            name = self._match_name(raw, others)
            if name and name not in matched:
                matched.append(name)

        minimum = 1 if cast is CastType.UNICAST else 2
        if len(matched) < minimum:
            pool = [n for n in others if n not in matched]
            self._rng.shuffle(pool)
            matched.extend(pool[: minimum - len(matched)])

        if cast is CastType.UNICAST:
            return (matched[0],)
        return tuple(matched)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _messages(self, view: AgentView, question: str) -> list:
        return [
            SystemMessage(content=self._system_prompt(view)),
            HumanMessage(content=self._situation(view, question)),
        ]

    def _system_prompt(self, view: AgentView) -> str:
        lines = [
            f"You are {view.self_name}, a player at the table in a game of Mafia (a.k.a. Werewolf), "
            "and your survival -- or your team's win -- genuinely depends on how you play this.",
            "",
            _ROLE_BRIEFS[view.role],
        ]
        if view.teammates:
            lines.append(f"Your fellow Mafia (you know each other on sight): {', '.join(view.teammates)}.")
        lines += [
            "",
            _PROTOCOL_BRIEF,
            "",
            _VOICE_BRIEF,
            "",
            "Stay in character at all times, and always name players exactly as given to "
            "you -- never invent, alter, or guess at a name.",
        ]
        return "\n".join(lines)

    def _situation(self, view: AgentView, question: str) -> str:
        lines = [
            f"Day {view.day_number} -- {view.phase.value} phase.",
            f"Still at the table, breathing: {', '.join(view.alive)}.",
        ]
        if view.dead:
            lines.append("Empty chairs -- gone, and everyone here knows it:")
            for record in view.dead:
                if record.revealed_role:
                    lines.append(
                        f"  - {record.name}: {record.cause} (Day {record.day_number}) -- "
                        f"unmasked on the spot as the {record.revealed_role}"
                    )
                else:
                    lines.append(
                        f"  - {record.name}: {record.cause} (Day {record.day_number}) -- "
                        "no one knows what they really were, and that's its own kind of haunting"
                    )
        if view.known_factions:
            known = ", ".join(f"{name} is {faction.value}" for name, faction in view.known_factions.items())
            lines.append(f"What you privately know, and only you know: {known}.")

        if self._memory:
            lines.append("")
            lines.append(
                "Your own running notes -- your private read on people and events, building "
                "as the game goes (lean on these; don't start from zero each time):"
            )
            lines.extend(f"  - {note}" for note in self._memory[-_MEMORY_LIMIT:])

        lines.append("")
        lines.append("The conversation so far, exactly as you've experienced it:")
        if view.feed:
            lines.extend(f"  {sighting.render()}" for sighting in view.feed)
        else:
            lines.append("  (nothing yet)")

        lines += ["", question]
        return "\n".join(lines)

    def _remember(self, view: AgentView, note: str) -> None:
        """File this turn's private reasoning away as a note-to-self for later turns.

        This is what gives an agent continuity of suspicion -- a read formed on
        Day 1 ("Drew dodged my question") can resurface and harden by Day 3
        ("...and now Drew's pushing hard to lynch the one person backing me up")
        instead of being silently re-derived, or lost, each time it's asked to act.
        """
        note = note.strip()
        if note:
            self._memory.append(f"({view.phase.value} {view.day_number}) {note}")

    # ------------------------------------------------------------------
    @staticmethod
    def _match_name(raw: str | None, candidates: list[str]) -> str | None:
        if not raw:
            return None
        cleaned = raw.strip().lower()
        for name in candidates:
            if name.lower() == cleaned:
                return name
        for name in candidates:
            if name.lower() in cleaned or cleaned in name.lower():
                return name
        return None

    def _warn(self, message: str) -> None:
        print(f"  [{self.name}] {message}")
