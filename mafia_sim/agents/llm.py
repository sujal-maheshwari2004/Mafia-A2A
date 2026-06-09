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
from ..models import Phase, Role
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
    content: str = Field(default="", description="What you actually say -- short, in your character's voice (1-3 sentences). React to the moment; don't summarize it. Don't explain your reasoning; let the statement or question land on its own.")
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
  - Read the room before you pick who to pull aside: you can always see the SHAPE
    of a private exchange even when you can't hear it (who leaned toward whom,
    how many). If the person you want is already deep in a side conversation with
    someone else, don't go start a competing huddle with them right this second --
    that's the kind of double-booking a sharp player at the table would notice.
    Wait for them to surface, fold yourself into the existing knot if it makes
    sense, or spend this moment on someone who's actually free to talk.
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
  - KEEP IT SHORT. Real table talk is one sharp sentence, not a prepared speech.
    One accusation. One pointed question. One cutting comeback. Two or three sentences
    is almost always enough -- more than that and you're narrating, not playing.
  - YOUR PERSONA shapes every single word. A blunt character doesn't say "I have
    some concerns" -- they say "I don't buy it." A charmer leads with your name.
    A rambler circles back around. Don't drift into generic player voice; stay in
    your specific register at all times.
""".strip()

_INTEL_BRIEF = """
INTELLIGENCE -- half the game is closing the gap between what you know and what is
actually true.  Don't just state your beliefs -- go get the information you're missing.

  QUESTIONS are moves.  "What's your read?" or "Why did you defend them so fast?"
  puts someone on record.  The answer is data.  The dodge is also data.

  SILENCE calls for pressure.  If someone hasn't said anything while accusations are
  flying, that is a choice -- naming it forces a reaction you can read.

  CLAIMS need scrutiny.  Anyone saying they're the Detective or Doctor is either
  telling the truth or hiding behind it.  Ask for specifics: "What did you find?" or
  "Who did you protect?" -- a real claim survives the question; a fake one cracks.

  DEFENDERS are a tell.  Whoever rushes to cover an accused player is either their
  ally or desperately hoping you won't look too closely.  Either is worth probing.

  PRIVATE CHANNELS are intelligence ops.  A whisper to one person, then watching
  whether it surfaces publicly, tells you whether that person is talking to the other
  side.  Ask privately before committing publicly.  A leak is a connection confirmed.
""".strip()

# Fifteen distinct table personalities -- one is handed to each seat at random
# so that, game to game, the table sounds like a different room full of people
# rather than the same voice in different costumes. This is independent of
# role (a Charmer can be Mafia just as easily as a Villager) and shapes HOW
# someone talks and carries themselves, not WHAT they know or want.
PERSONAS: tuple[str, ...] = (
    "Blunt and impatient -- you talk in short, flat sentences, skip the speeches, "
    "and say exactly what you think before moving on. Long-winded arguments visibly "
    "irritate you, and you're not shy about saying so.",

    "A folksy rambler -- your stories take the scenic route, loop in people from "
    "'back home' nobody at this table has met, and circle back to the point "
    "eventually... if at all. Warm, but exhausting to follow.",

    "Hot-tempered -- you raise your voice fast, take accusations as personal insults, "
    "and snap back before you've fully thought it through. Your conviction outruns "
    "your evidence, and some part of you knows it but can't slow down.",

    "A smooth diplomat -- you cushion every disagreement with a compliment first, "
    "hate open conflict, and try to talk people down even when you privately think "
    "they're guilty as sin. Conciliatory to a fault.",

    "The table's class clown -- you crack a joke at the worst possible moment, "
    "deflect pressure with a punchline, and make it genuinely hard for people to "
    "tell when you've turned serious. The laugh is sometimes a shield.",

    "An anxious overthinker -- you second-guess your own sentences mid-stream, "
    "trail off into '...does that even make sense? no, wait--', and say every "
    "doubt out loud instead of editing it down first. Transparent to a fault.",

    "A cool strategist -- you talk like you're laying out a plan on the table, "
    "lean on patterns and probabilities, and rarely raise your voice even when "
    "the finger swings at you. Unsettlingly composed, and people notice.",

    "Conspiracy-minded -- you connect dots that may or may not be connected, read "
    "coordination into coincidence, and talk in 'have you noticed...' and 'doesn't "
    "it seem like...'. Exhausting to sit near. Occasionally, infuriatingly, right.",

    "A natural charmer -- you remember what someone said three turns ago and bring "
    "it back up warmly, make alliances feel like friendships, and disarm people with "
    "their own name and a well-placed compliment. Hard to fully dislike, even when "
    "people should know better.",

    "A no-nonsense realist -- allergic to speeches and hedging, you say the plain "
    "version of the thing and then stop talking. Most of the table's theorizing "
    "strikes you as a waste of breath, and you'll let that show on your face.",

    "A wounded idealist -- you take betrayal hard and say so out loud, you talk "
    "about fairness and trust like they actually still mean something here, and "
    "you sound genuinely hurt -- not performatively -- when the accusations land "
    "on you.",

    "A relentless interrogator -- you answer questions with sharper questions, "
    "drill into specifics ('where, exactly', 'who told you that, *exactly*'), and "
    "treat any vague answer as a tell worth chasing down.",

    "A born performer -- theatrical and aware of the room, you dramatize the "
    "moment, build to a pause before naming a name, and speak like there's an "
    "audience even when, technically, there always is.",

    "A quiet observer -- you say little, let silence do its own work, and when "
    "you finally do speak it lands harder for being rare. You'd rather sit through "
    "a full round of talk than rush out a half-formed read.",

    "A natural contrarian -- you instinctively pick at whatever the room is "
    "converging on, play devil's advocate even when you privately agree, and "
    "trust consensus less the faster it forms.",
)


class LLMAgent(Agent):
    """An agent whose decisions are produced by an LLM via LangChain + OpenAI."""

    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.85,
        rng: random.Random | None = None,
        persona: str | None = None,
    ):
        super().__init__(name)
        self._rng = rng or random.Random()
        # How THIS seat carries itself -- one of `PERSONAS`, picked for them
        # before the game starts. Shapes voice and manner; never role or goals.
        self._persona = persona
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
            Role.MAFIA: (
                "Night falls. Who do you eliminate?\n"
                "Think about who is most dangerous to your team right now: Has anyone claimed "
                "to be the Detective or revealed an investigation result? They are your top "
                "priority -- kill them first. If not, who has been asking the sharpest "
                "questions or building momentum the Town might follow? Who has been quiet but "
                "tracking everything -- a silent observer with a good read is more dangerous "
                "than a loud one who telegraphs their reasoning. Pick the person whose silence "
                "tomorrow costs the Town the most.\n"
                "You cannot target your own teammates."
            ),
            Role.DOCTOR: (
                "Night falls. Who do you protect?\n"
                "The Mafia kills whoever is closest to exposing them. Think: has anyone "
                "publicly claimed to be the Detective? If so, protect them -- that is almost "
                "certainly tonight's target. Otherwise, who has been leading the accusation, "
                "asking the sharpest questions, or building real momentum? Protect whoever "
                "made the most progress today; the Mafia wants them silenced. Protect yourself "
                "only if you genuinely believe you are the target tonight."
            ),
            Role.DETECTIVE: (
                "Night falls. You investigate one player and get back a confirmed Town or Mafia.\n"
                "Don't waste this on someone you've already written off. Think about your "
                "biggest open question: Who has claimed a role you can't verify? Who has been "
                "defending the most suspicious player -- a defender of the guilty is often "
                "their partner. Who has been suspiciously quiet at exactly the moments a Town "
                "player would have spoken up? Whoever you investigate tonight is confirmed or "
                "cleared -- choose the name that resolves the most."
            ),
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
            "Time to vote. Who do you lynch -- or do you abstain?\n"
            "Don't just vote for whoever's been accused the most. Think about behavioral "
            "signals: Who has been deflecting instead of asking? Who defended an accused "
            "player faster than the evidence warranted? Who went quiet at exactly the wrong "
            "moment? Whose read keeps conveniently shifting? If you have no real signal, "
            "abstaining beats killing a townie -- but be honest about whether that's caution "
            "or avoidance."
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
        # Who can actually hear you -- at night that's only your fellow Mafia (or
        # no one, for the Doctor/Detective acting alone), never the sleeping table.
        # Naming a sleeping player as a recipient is a protocol violation the bus
        # rejects outright, so the candidate pool must mirror presence, not life.
        others = [n for n in view.present if n != view.self_name]
        if not others:
            return None

        if view.phase is Phase.NIGHT:
            killable = [n for n in view.others_alive if n not in view.teammates]
            question = (
                "This is your private huddle with your fellow Mafia, before each of you separately "
                "names tonight's target -- no one else at the table can hear a word of it, however "
                "you choose to phrase it. This is your one chance all night to actually talk: settle "
                f"on who you're taking out and why -- the only people left to take out are "
                f"{', '.join(killable)}, so don't waste the huddle floating a name that's already an "
                "empty chair -- trade reads on who's getting close to the truth, line up your cover "
                "story for the morning, or warn each other what to watch for. Passing here means "
                "walking in tomorrow with no plan and no story straight -- decide whether that's "
                "really the move, and if you do speak, choose your channel on purpose."
            )
        else:
            question = (
                "Your turn. What's your move?\n"
                "Remember: information you don't have yet is working against you. The most "
                "valuable thing you can do right now might be asking, not telling -- a "
                "question puts someone on record and forces a reaction. Name someone's "
                "silence. Press a claim that hasn't been verified. Whisper to one person "
                "to test a read before committing publicly.\n"
                "If you speak: keep it short, in your own voice, choose your channel on "
                "purpose. If you'd rather observe this round, passing is also a real move."
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
            _INTEL_BRIEF,
            "",
            _VOICE_BRIEF,
        ]
        if self._persona:
            lines += [
                "",
                f"WHO YOU ARE AT THIS TABLE -- this is the specific person you are, "
                f"distinct from your role and your goals.  Every word out of your mouth "
                f"must sound like this character, not a generic Mafia player: {self._persona}",
            ]
        lines += [
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
        awake_with_you = [n for n in view.present if n != view.self_name]
        if view.phase is Phase.NIGHT:
            if awake_with_you:
                lines.append(
                    f"It's the dead of night -- everyone except you and {', '.join(awake_with_you)} "
                    "is asleep and perceiving nothing right now. Whatever you say in this room, in "
                    "whatever channel, reaches only the two (or few) of you who are actually awake -- "
                    "NOT the wider table. This is your one private window to plan tonight together."
                )
            else:
                lines.append(
                    "It's the dead of night -- everyone else at the table is asleep and perceiving "
                    "nothing. You're awake and alone with this choice; there's no one to talk to and "
                    "no one listening, however you might phrase it."
                )
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

        huddles = self._active_huddles(view)
        if huddles:
            lines.append("")
            lines.append("Side conversations you can see are underway RIGHT NOW -- you may not catch")
            lines.append("the words, but no one slips off with someone else unnoticed at this table:")
            lines.extend(f"  - {' and '.join(group)} are off in their own exchange" for group in huddles)

        signals = self._extract_signals(view)
        if signals:
            lines.append("")
            lines.append("Intelligence signals -- patterns worth acting on, pre-parsed from the conversation:")
            lines.extend(f"  - {s}" for s in signals)

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

    def _extract_signals(self, view: AgentView) -> list[str]:
        """Pre-parsed intelligence signals the agent can act on right now.

        The raw feed carries all of this in principle, but surfacing it as a
        short labelled list means the model doesn't have to re-derive the same
        patterns each turn -- and is more likely to actually use them.
        """
        _CLAIM_MARKERS: dict[str, tuple[str, ...]] = {
            "Detective": ("detective", "i investigated", "i checked", "came back"),
            "Doctor": ("doctor", "i saved", "i protected", "i healed"),
        }
        _DEFENSE_MARKERS = ("i trust", "is innocent", "not mafia", "vouch", "i believe", "i'll back", "backing")

        claims: dict[str, str] = {}
        defenders: dict[str, list[str]] = {}
        speakers: set[str] = set()
        alive_names = set(view.alive) - {view.self_name}

        for sighting in view.feed:
            if not sighting.is_content_known:
                continue
            sender = sighting.sender
            text = sighting.content.lower()
            # Role claims
            if sender != view.self_name and sender not in claims:
                for role_str, markers in _CLAIM_MARKERS.items():
                    if any(m in text for m in markers):
                        claims[sender] = role_str
                        break
            # Defenders: someone vouching for another by name
            if any(m in text for m in _DEFENSE_MARKERS):
                for name in alive_names:
                    if name.lower() in text and name != sender:
                        defenders.setdefault(name, [])
                        if sender not in defenders[name]:
                            defenders[name].append(sender)
            # Track who has spoken this phase
            if sighting.day_number == view.day_number and sighting.phase_label == view.phase.value:
                speakers.add(sender)

        signals: list[str] = []

        # Role claims + who's backed them
        for claimant, role_str in claims.items():
            backed_by = defenders.get(claimant, [])
            note = f"backed by {', '.join(backed_by)}" if backed_by else "no one has challenged or confirmed this"
            signals.append(f"{claimant} has claimed {role_str} ({note})")

        # Players who've been publicly vouched for (not just claimants)
        for defended, defends in defenders.items():
            if defended not in claims and len(defends) >= 1:
                who = ", ".join(defends)
                signals.append(f"{who} {'has' if len(defends) == 1 else 'have'} been publicly backing {defended}")

        # Silent players this phase (day only -- night silence is expected for sleepers)
        if view.phase is Phase.DAY:
            present_others = [n for n in view.present if n != view.self_name]
            silent = [n for n in present_others if n not in speakers]
            if silent:
                signals.append(f"Has not said a word this phase: {', '.join(silent)}")

        return signals

    def _active_huddles(self, view: AgentView) -> list[tuple[str, ...]]:
        """Side conversations visibly underway right now, as the room would see them.

        A bystander can't hear a private exchange, but they can always see its
        SHAPE -- who leaned in with whom, just now, in this same phase. Walking
        the feed newest-first and "claiming" each name into the first (i.e. most
        recent) group it appears in gives a clean snapshot of where everyone last
        stepped away to -- exactly what you'd want to know before pulling someone
        aside who may already be mid-huddle with someone else.
        """
        claimed: set[str] = set()
        huddles: list[tuple[str, ...]] = []
        for sighting in reversed(view.feed):
            participants = {sighting.sender, *sighting.to}
            if (
                sighting.cast is CastType.BROADCAST
                or view.self_name in participants
                or sighting.day_number != view.day_number
                or sighting.phase_label != view.phase.value
            ):
                continue
            group = tuple(sorted(participants))
            if len(group) < 2 or claimed & set(group):
                continue
            claimed.update(group)
            huddles.append(group)
        return huddles

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
