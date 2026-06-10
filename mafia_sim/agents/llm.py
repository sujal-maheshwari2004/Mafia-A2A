"""LLM-backed agent (LangChain + OpenAI).

Implements the exact same `Agent` interface as `HeuristicAgent` -- the comm
protocol and the simulation orchestrator are completely agent-agnostic, so
this is a drop-in swap (and any future agent, on any stack, plugs in the same
way). It reasons solely from its `AgentView`: a system prompt explains its
identity, the rules, and the A2A protocol; a per-turn human prompt supplies
the live situation -- who's alive, what it has personally perceived -- and
the question at hand.

Each distinct game action is exposed as a **tool** the model can call, rather
than a flat schema it must fill in.  Nine tools cover every action:

  Night (role-specific):  _EliminateTool  _ProtectTool  _InvestigateTool
  Vote:                   _VoteTool  _AbstainTool
  Discussion:             _BroadcastTool  _WhisperTool  _HuddleTool  _PassTurnTool

The model picks the right tool for its intent; `tool_choice="any"` forces a
decision for night and vote; discussion is open (no call = stay silent).

Before each action, a separate cheap "table read" call (`_TableRead`, run on
`_EXTRACTION_MODEL`) reads the conversation contextually -- who's actually
under suspicion, who's been defended, role claims, contradictions, and (at
night) whether the Mafia have converged on a kill target. This replaces
keyword/substring scanning of the feed: it's the difference between matching
the word "mafia" in "X is NOT mafia" and understanding that the sentence
clears X.

Requires `OPENAI_API_KEY` in the environment (langchain-openai reads it
automatically).
"""

from __future__ import annotations

import difflib
import random
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..budget import CallBudget
from ..protocol import CastType, CommRequest, Sighting
from ..models import Faction, Phase, Role
from .base import Agent, AgentView

DEFAULT_MODEL = "gpt-4.1-mini"

# Used for the "table read" pass only -- a cheap, contextual reading of the
# conversation (who's under suspicion, who's been defended, etc.) that replaces
# keyword/substring scanning. Runs once per turn alongside the main call, so it
# needs to be fast and inexpensive rather than the most capable model available.
_EXTRACTION_MODEL = "gpt-4.1-nano"

# How many of an agent's own past private notes get fed back into its next
# prompt -- enough continuity to remember a running suspicion without letting
# the prompt grow without bound over a long game.
_MEMORY_LIMIT = 14

# Cap on feed entries injected into each prompt -- prevents the model from
# pattern-matching on a wall of near-identical messages in long games.
_FEED_WINDOW = 25

# Network defaults for every ChatOpenAI client below: a hung call must not be
# allowed to block the worker thread (and the whole game) forever, but a
# couple of quick retries absorb transient network blips before falling back
# to the existing per-call fallbacks (random choice / silence).
_REQUEST_TIMEOUT = 60  # seconds
_MAX_RETRIES = 2

_REASONING_FIELD = (
    "Your own private take on this moment -- who you suspect and why, what you're "
    "tracking, what you plan to do about it. Never shown to anyone else, but it IS "
    "kept as your own running notes for the rest of the game, so write it the way "
    "you'd actually want to be reminded of your own read later: specific, and honest "
    "about your suspicions, doubts, and hunches -- not a tidy summary for an audience."
)


# ── Table read: a contextual reading of the conversation, produced by a small ──
# ── model, that replaces keyword/substring scanning of the feed. ──────────────

class _PlayerNote(BaseModel):
    name: str = Field(description="Exact player name")
    note: str = Field(description="One short sentence: what's notable about them and why")


class _TableRead(BaseModel):
    """A contextual analysis of the table talk so far, in place of keyword matching."""

    under_suspicion: list[_PlayerNote] = Field(
        default_factory=list,
        description=(
            "Players currently under genuine suspicion based on what's actually been said, "
            "and why. Do NOT include someone here if they were only mentioned while being "
            "DEFENDED, cleared, or vouched for -- a sentence that clears someone is the "
            "OPPOSITE of an accusation against them, even if it contains words like "
            "'mafia' or 'suspicious'."
        ),
    )
    defended: list[_PlayerNote] = Field(
        default_factory=list,
        description="Players who've been vouched for, defended, or cleared, and by whom.",
    )
    role_claims: list[_PlayerNote] = Field(
        default_factory=list,
        description=(
            "Players who've claimed to be the Detective or Doctor, whether the claim has "
            "been challenged, and whether it holds up under scrutiny."
        ),
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description=(
            "Specific inconsistencies worth pointing out: a story that changed, a vote "
            "that flipped without explanation, someone defending a player who later "
            "looked guilty, etc. Empty list if nothing stands out."
        ),
    )
    agreed_night_target: str | None = Field(
        default=None,
        description=(
            "ONLY relevant for a Mafia night huddle: if the team has clearly converged on "
            "who to eliminate tonight, name that player exactly. If there's no clear "
            "agreement yet, or this isn't a Mafia night huddle, return null. A message "
            "arguing AGAINST targeting someone does not count as agreement on that person."
        ),
    )


# ── Day recap: a compact per-day digest, generated once a day has fully ───────
# ── scrolled past, so early-game context survives long games. ─────────────────

class _DaySummary(BaseModel):
    """A compact private recap of one day, from one player's point of view."""

    summary: str = Field(
        description=(
            "A compact 1-2 sentence private recap of this day, written for the player's "
            "own future reference: key accusations or claims made, the lynch outcome, and "
            "anything that still feels relevant going forward."
        )
    )


# ── Night-action tools (each role sees only the one that applies to them) ─────

class _EliminateTool(BaseModel):
    """[MAFIA] Commit to eliminating this player tonight."""
    target: str = Field(description="Exact name of the player to eliminate")
    reasoning: str = Field(description=_REASONING_FIELD)


class _ProtectTool(BaseModel):
    """[DOCTOR] Choose one player to shield from tonight's Mafia kill."""
    target: str = Field(description="Exact name of the player to protect")
    reasoning: str = Field(description=_REASONING_FIELD)


class _InvestigateTool(BaseModel):
    """[DETECTIVE] Investigate one player and learn whether they are Town or Mafia."""
    target: str = Field(description="Exact name of the player to investigate")
    reasoning: str = Field(description=_REASONING_FIELD)


# ── Vote tools ────────────────────────────────────────────────────────────────

class _VoteTool(BaseModel):
    """Cast your lynch vote for a specific player."""
    target: str = Field(description="Exact name of the player you vote to lynch")
    reasoning: str = Field(description=_REASONING_FIELD)


class _AbstainTool(BaseModel):
    """Pass on this vote -- contribute nothing to the tally."""
    reasoning: str = Field(description=_REASONING_FIELD)


# ── Discussion tools ──────────────────────────────────────────────────────────

class _BroadcastTool(BaseModel):
    """Speak to everyone currently at the table."""
    content: str = Field(description="What you say -- short, in your own voice (1-3 sentences). React to the moment; don't explain your reasoning.")
    reasoning: str = Field(description=_REASONING_FIELD)


class _WhisperTool(BaseModel):
    """Lean over and privately say something to exactly one person. Everyone else can see you whispered but not what you said."""
    recipient: str = Field(description="Exact name of the single person you are whispering to")
    content: str = Field(description="What you say -- short, in your own voice (1-3 sentences). React to the moment; don't explain your reasoning.")
    reasoning: str = Field(description=_REASONING_FIELD)


class _HuddleTool(BaseModel):
    """Pull two or more specific people into a private side conversation. Everyone else sees who huddled but not what was said."""
    recipients: list[str] = Field(description="Exact names of 2 or more people in the huddle")
    content: str = Field(description="What you say -- short, in your own voice (1-3 sentences). React to the moment; don't explain your reasoning.")
    reasoning: str = Field(description=_REASONING_FIELD)


class _PassTurnTool(BaseModel):
    """Stay silent this turn -- observe instead of committing to a message."""
    reasoning: str = Field(description=_REASONING_FIELD)


_ROLE_BRIEFS = {
    Role.MAFIA: (
        "You are MAFIA. Each night you and your teammates secretly choose one player to eliminate. "
        "You win once the Mafia equal or outnumber the Town. Blend in, deflect suspicion, and "
        "steer the Town's votes toward townsfolk -- especially anyone who seems to be figuring you out. "
        "CRITICAL: You must NEVER vote to lynch one of your own Mafia teammates. Doing so hands the "
        "Town a free win. Always vote for a Town player, or abstain if needed."
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
        budget: CallBudget | None = None,
    ):
        super().__init__(name)
        self._rng = rng or random.Random()
        # Shared per-game ceiling on LLM activity -- once exhausted, every
        # decision below falls back to its existing cheap default instead of
        # calling the model at all.
        self._budget = budget
        # How THIS seat carries itself -- one of `PERSONAS`, picked for them
        # before the game starts. Shapes voice and manner; never role or goals.
        self._persona = persona
        # Continuity of suspicion: every private "reasoning" the model writes
        # gets kept here and re-served back to it next turn, so a read formed
        # on Day 1 can still shape a vote on Day 3 instead of being re-derived
        # -- or quietly forgotten -- from scratch each time.
        self._memory: list[str] = []
        # Keyed by "day:phase" -- what this agent actually said each phase.
        # Injected back into every prompt so the model doesn't keep re-sending
        # the same message (the 12-confirmation night-coordination problem).
        self._phase_transcript: dict[str, list[str]] = {}
        # Keyed by "day:phase" -- who this agent has whispered/huddled with
        # this phase, so it can be nudged to stop re-asking the same people
        # the same thing.
        self._probed_this_phase: dict[str, set[str]] = {}
        # day_number -> compact recap, generated once that day is no longer
        # "today" -- keeps early-game context alive without bloating the
        # raw feed shown each turn.
        self._day_summaries: dict[int, str] = {}
        llm = ChatOpenAI(
            model=model, temperature=temperature, timeout=_REQUEST_TIMEOUT, max_retries=_MAX_RETRIES
        )
        # Each role sees only the single night-action tool that applies to them;
        # tool_choice="any" forces the model to always call it.
        self._night_llm: dict[Role, object] = {
            Role.MAFIA:      llm.bind_tools([_EliminateTool],   tool_choice="any"),
            Role.DOCTOR:     llm.bind_tools([_ProtectTool],     tool_choice="any"),
            Role.DETECTIVE:  llm.bind_tools([_InvestigateTool], tool_choice="any"),
        }
        # Vote: must call exactly one of these two tools.
        self._vote_llm = llm.bind_tools([_VoteTool, _AbstainTool], tool_choice="any")
        # Discussion: model may call one tool or none (= stay silent).
        self._discussion_llm = llm.bind_tools(
            [_BroadcastTool, _WhisperTool, _HuddleTool, _PassTurnTool]
        )
        # Cheap, separate model for the "table read" -- a contextual pass over
        # the conversation that replaces keyword/substring scanning. Temperature
        # 0 because this is meant to be a faithful read, not a creative one.
        self._table_read_llm = ChatOpenAI(
            model=_EXTRACTION_MODEL, temperature=0, timeout=_REQUEST_TIMEOUT, max_retries=_MAX_RETRIES
        ).with_structured_output(_TableRead)
        # Same cheap model, bound for the once-per-day recap instead.
        self._day_summary_llm = ChatOpenAI(
            model=_EXTRACTION_MODEL, temperature=0, timeout=_REQUEST_TIMEOUT, max_retries=_MAX_RETRIES
        ).with_structured_output(_DaySummary)

    # ------------------------------------------------------------------
    # Night
    # ------------------------------------------------------------------
    def choose_night_target(self, view: AgentView) -> str:
        candidates = self._night_candidates(view)
        if self._budget is not None:
            if self._budget.exhausted:
                return self._rng.choice(candidates)
            self._budget.record()
        table_read = self._get_table_read(view)
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
            response = self._night_llm[view.role].invoke(self._messages(view, question, table_read))
        except Exception as exc:
            self._warn(f"night-action call failed ({exc!r}); choosing at random")
            return self._rng.choice(candidates)

        if not response.tool_calls:
            return self._rng.choice(candidates)
        tc = response.tool_calls[0]["args"]
        self._remember(view, tc.get("reasoning", ""))
        return self._match_name(tc.get("target", ""), candidates) or self._rng.choice(candidates)

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
        if self._budget is not None:
            if self._budget.exhausted:
                return None  # abstain
            self._budget.record()

        table_read = self._get_table_read(view)
        if table_read.under_suspicion:
            pressure_str = "\n".join(f"  - {p.name}: {p.note}" for p in table_read.under_suspicion)
        else:
            pressure_str = "  (no one stood out as a clear accusation target today)"
        question = (
            f"Time to vote. Here's a contextual read of today's discussion:\n{pressure_str}\n\n"
            "This is real signal -- the room built it over this whole day phase. "
            "Don't throw it away. Vote for the player the discussion actually pointed at most "
            "strongly, unless you have a specific reason to believe that momentum was manufactured.\n\n"
            "That said, don't ONLY follow the read above: Who has been deflecting instead of asking? "
            "Who defended an accused player faster than the evidence warranted? "
            "Whose read keeps conveniently shifting? "
            "If you have no real signal at all, abstaining beats killing a townie -- "
            "but be honest about whether that's caution or avoidance."
        )
        try:
            response = self._vote_llm.invoke(self._messages(view, question, table_read))
        except Exception as exc:
            self._warn(f"vote call failed ({exc!r}); abstaining")
            return None

        if not response.tool_calls:
            return None
        tc = response.tool_calls[0]
        self._remember(view, tc["args"].get("reasoning", ""))
        if tc["name"] == "_AbstainTool":
            return None
        return self._match_name(tc["args"].get("target", ""), others)

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
        if self._budget is not None:
            if self._budget.exhausted:
                return None  # stay silent
            self._budget.record()

        table_read = self._get_table_read(view)

        if view.phase is Phase.NIGHT:
            killable = [n for n in view.others_alive if n not in view.teammates]
            agreed_target = self._match_name(table_read.agreed_night_target, killable)
            if agreed_target:
                question = (
                    f"Your team has already agreed to eliminate {agreed_target} tonight -- "
                    "the decision is made, repeating it adds nothing. "
                    "If you speak now, spend it on something useful: your cover story for tomorrow, "
                    "who to redirect suspicion onto, what to watch for in the morning discussion, "
                    "or who the next target should be. "
                    "If there's nothing new to say, stay silent (speak=false) -- "
                    "a quiet night is better than confirming the same name a fifth time."
                )
            else:
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
            if view.role is Role.DETECTIVE:
                unclaimed = [
                    name for name, faction in view.known_factions.items()
                    if faction is Faction.MAFIA and not self._has_claimed(name)
                ]
                if unclaimed:
                    question = (
                        f"URGENT -- you privately know that {', '.join(unclaimed)} is Mafia, "
                        "and the Town doesn't know it yet. Every phase you stay quiet is a "
                        "phase the Mafia gets to operate freely, and if you're killed tonight "
                        "this evidence dies with you. Seriously weigh claiming Detective and "
                        "naming what you found against the risk of becoming tonight's target "
                        "for staying silent.\n\n"
                    ) + question
        try:
            response = self._discussion_llm.invoke(self._messages(view, question, table_read))
        except Exception as exc:
            self._warn(f"discussion call failed ({exc!r}); staying silent")
            return None

        if not response.tool_calls:
            return None  # model chose not to act this turn

        tc = response.tool_calls[0]
        self._remember(view, tc["args"].get("reasoning", ""))

        name = tc["name"]

        if name == "_PassTurnTool":
            return None

        content = tc["args"].get("content", "").strip()
        if not content:
            return None

        if name == "_BroadcastTool":
            cast, to = CastType.BROADCAST, ()

        elif name == "_WhisperTool":
            recipient = self._match_name(tc["args"].get("recipient", ""), others)
            if not recipient:
                recipient = self._rng.choice(others)
            cast, to = CastType.UNICAST, (recipient,)

        elif name == "_HuddleTool":
            raw = tc["args"].get("recipients", [])
            matched: list[str] = []
            for r in raw:
                m = self._match_name(r, others)
                if m and m not in matched:
                    matched.append(m)
            # Need at least 2 for a valid multicast; pad from remaining if short
            if len(matched) < 2:
                pool = [n for n in others if n not in matched]
                self._rng.shuffle(pool)
                matched.extend(pool[: 2 - len(matched)])
            if len(matched) < 2:
                cast, to = CastType.BROADCAST, ()
            else:
                cast, to = CastType.MULTICAST, tuple(matched)

        else:
            return None  # unknown tool -- stay silent

        phase_key = f"{view.day_number}:{view.phase.value}"
        if self._is_repeat(content, self._phase_transcript.get(phase_key, [])):
            return None  # near-duplicate of something already said this phase -- stay silent

        if cast is not CastType.BROADCAST:
            self._probed_this_phase.setdefault(phase_key, set()).update(to)

        self._phase_transcript.setdefault(phase_key, []).append(content)
        return CommRequest(cast, to, content)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _messages(self, view: AgentView, question: str, table_read: _TableRead) -> list:
        return [
            SystemMessage(content=self._system_prompt(view)),
            HumanMessage(content=self._situation(view, question, table_read)),
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

    def _situation(self, view: AgentView, question: str, table_read: _TableRead) -> str:
        self._ensure_day_summaries(view)
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

        if view.vote_history:
            lines.append("")
            lines.append("Lynch vote history:")
            for day, votes in sorted(view.vote_history.items()):
                if not votes:
                    continue
                cast_strs = [
                    f"{voter} -> {target}" if target else f"{voter} -> abstain"
                    for voter, target in votes.items()
                ]
                suffix = " (so far)" if day == view.day_number and view.phase is Phase.DAY else ""
                lines.append(f"  Day {day}{suffix}: " + ", ".join(cast_strs))

        huddles = self._active_huddles(view)
        if huddles:
            lines.append("")
            lines.append("Side conversations you can see are underway RIGHT NOW -- you may not catch")
            lines.append("the words, but no one slips off with someone else unnoticed at this table:")
            lines.extend(f"  - {' and '.join(group)} are off in their own exchange" for group in huddles)

        if (
            table_read.under_suspicion
            or table_read.defended
            or table_read.role_claims
            or table_read.contradictions
        ):
            lines.append("")
            lines.append("Table read -- a contextual reading of the conversation so far:")
            for p in table_read.under_suspicion:
                lines.append(f"  - SUSPICION: {p.name} -- {p.note}")
            for p in table_read.defended:
                lines.append(f"  - DEFENDED: {p.name} -- {p.note}")
            for p in table_read.role_claims:
                lines.append(f"  - ROLE CLAIM: {p.name} -- {p.note}")
            for c in table_read.contradictions:
                lines.append(f"  - CONTRADICTION: {c}")

        if view.phase is Phase.DAY:
            silent = self._silent_players(view)
            if silent:
                lines.append("")
                lines.append(f"Has not said a word this phase: {', '.join(silent)}")

        if self._memory:
            lines.append("")
            lines.append(
                "Your own running notes -- your private read on people and events, building "
                "as the game goes (lean on these; don't start from zero each time):"
            )
            lines.extend(f"  - {note}" for note in self._memory[-_MEMORY_LIMIT:])

        phase_key = f"{view.day_number}:{view.phase.value}"
        own_msgs = self._phase_transcript.get(phase_key, [])
        if own_msgs:
            lines.append("")
            lines.append(
                "What YOU have already said this phase -- each of these has been heard. "
                "Do NOT repeat or rephrase any of them. If you'd say the same thing again, "
                "stay silent instead (speak=false). Say something new or say nothing:"
            )
            lines.extend(f"  - \"{msg}\"" for msg in own_msgs[-6:])

        probed = self._probed_this_phase.get(phase_key)
        if probed:
            lines.append("")
            lines.append(
                f"You've already pulled these people aside privately this phase: "
                f"{', '.join(sorted(probed))}. Repeating the same question to them will feel "
                "robotic -- bring something new, wait for their answer, or turn your "
                "attention to someone else."
            )

        earlier_days = {day: summary for day, summary in sorted(self._day_summaries.items()) if summary}
        if earlier_days:
            lines.append("")
            lines.append("Earlier days, in brief (the full transcript has scrolled past):")
            lines.extend(f"  Day {day}: {summary}" for day, summary in earlier_days.items())

        lines.append("")
        lines.append("Today's conversation so far, exactly as you've experienced it:")
        feed = [s for s in view.feed if s.day_number == view.day_number]
        if len(feed) > _FEED_WINDOW:
            lines.append(f"  ... ({len(feed) - _FEED_WINDOW} earlier messages omitted) ...")
            feed = feed[-_FEED_WINDOW:]
        if feed:
            lines.extend(f"  {sighting.render()}" for sighting in feed)
        else:
            lines.append("  (nothing yet)")

        lines += ["", question]
        return "\n".join(lines)

    def _silent_players(self, view: AgentView) -> list[str]:
        """Players present this phase who haven't sent anything you could perceive yet.

        Purely structural -- based on who has a sighting in this phase, not on
        what anyone said -- so it stays accurate regardless of phrasing.
        """
        speakers: set[str] = set()
        for sighting in view.feed:
            if sighting.day_number == view.day_number and sighting.phase_label == view.phase.value:
                speakers.add(sighting.sender)
        present_others = [n for n in view.present if n != view.self_name]
        return [n for n in present_others if n not in speakers]

    def _get_table_read(self, view: AgentView) -> _TableRead:
        """A cheap, contextual read of the conversation so far -- replaces keyword scanning."""
        if not view.feed:
            return _TableRead()
        try:
            result = self._table_read_llm.invoke(self._table_read_prompt(view))
        except Exception as exc:
            self._warn(f"table-read call failed ({exc!r}); skipping contextual signals")
            return _TableRead()
        return result if isinstance(result, _TableRead) else _TableRead()

    def _table_read_prompt(self, view: AgentView) -> list:
        lines = [
            "You are an impartial observer analyzing the table talk in a game of Mafia (Werewolf), "
            f"from the point of view of what the player {view.self_name} has personally seen or heard.",
            f"Players still at the table: {', '.join(view.alive)}.",
        ]
        if view.phase is Phase.NIGHT and view.teammates:
            lines.append(
                f"This is a private Mafia night huddle. {view.self_name}'s fellow Mafia: "
                f"{', '.join(view.teammates)}."
            )
        lines.append("")
        lines.append("Conversation, oldest first:")
        feed = list(view.feed)
        if len(feed) > _FEED_WINDOW:
            feed = feed[-_FEED_WINDOW:]
        if feed:
            lines.extend(f"  {sighting.render()}" for sighting in feed)
        else:
            lines.append("  (nothing yet)")
        lines.append("")
        lines.append(
            "Read this carefully and report what's ACTUALLY going on -- not which words were used. "
            "A sentence that defends, clears, or vouches for someone is the OPPOSITE of an "
            "accusation against them, even if it contains words like 'suspicious' or 'mafia'. "
            "Only report a Mafia night-kill agreement if the team has clearly converged on one "
            "name; arguing against a name is not agreement on it."
        )
        return [HumanMessage(content="\n".join(lines))]

    def _ensure_day_summaries(self, view: AgentView) -> None:
        """Generate and cache a recap for any day that's no longer "today".

        Called every turn but cheap after the first call for a given day:
        once `self._day_summaries[day]` is set (even to `""`, for a quiet
        day with nothing perceived), it's never recomputed.
        """
        by_day: dict[int, list[Sighting]] = {}
        for sighting in view.feed:
            if sighting.day_number < view.day_number:
                by_day.setdefault(sighting.day_number, []).append(sighting)
        for day, sightings in by_day.items():
            if day not in self._day_summaries:
                self._day_summaries[day] = self._summarize_day(view, day, sightings)

    def _summarize_day(self, view: AgentView, day: int, sightings: list[Sighting]) -> str:
        """A compact 1-2 sentence recap of `day`, from `view.self_name`'s point of view."""
        lines = [
            f"You are recapping Day {day} of a game of Mafia (Werewolf) from "
            f"{view.self_name}'s point of view, for their own future reference -- the "
            "detailed transcript is about to scroll out of view.",
            "",
            "What they personally perceived that day:",
        ]
        lines.extend(f"  {sighting.render()}" for sighting in sightings)
        votes = view.vote_history.get(day)
        if votes:
            cast_strs = [
                f"{voter} -> {target}" if target else f"{voter} -> abstain"
                for voter, target in votes.items()
            ]
            lines.append("")
            lines.append(f"Lynch votes that day: {', '.join(cast_strs)}")
        lines.append("")
        lines.append(
            "Write a compact 1-2 sentence private recap: key accusations or claims made, "
            "the lynch outcome, and anything that still feels relevant going forward."
        )
        try:
            result = self._day_summary_llm.invoke([HumanMessage(content="\n".join(lines))])
        except Exception as exc:
            self._warn(f"day-summary call failed ({exc!r}); skipping")
            return ""
        return result.summary.strip() if isinstance(result, _DaySummary) else ""

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
    def _is_repeat(content: str, recent: list[str], threshold: float = 0.82) -> bool:
        """Is `content` a near-duplicate of something already said this phase?

        Catches rephrasings the model might slip past its own "don't repeat
        yourself" instruction -- e.g. asking the same person the same
        question with slightly different wording.
        """
        lowered = content.lower()
        return any(
            difflib.SequenceMatcher(None, lowered, prev.lower()).ratio() >= threshold
            for prev in recent
        )

    def _has_claimed(self, name: str) -> bool:
        """Has this agent ever publicly named `name` in something it said?

        A cheap proxy for "have I already raised this" -- used to avoid
        nagging a Detective who's already started pointing at their suspect.
        """
        lowered = name.lower()
        return any(
            lowered in msg.lower()
            for msgs in self._phase_transcript.values()
            for msg in msgs
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _match_name(raw: str | None, candidates: list[str]) -> str | None:
        """Resolve a model-supplied name to one of `candidates`.

        Exact match first, then a word-boundary match (handles trailing
        punctuation or a name embedded in a longer phrase like "Drew did
        it"), then a thresholded fuzzy match (handles minor typos). Never
        falls back to a raw substring check -- that would let a short name
        like "Kai" match "Kaiden" or vice versa.
        """
        if not raw:
            return None
        cleaned = raw.strip().lower()
        lowered = {name.lower(): name for name in candidates}

        if cleaned in lowered:
            return lowered[cleaned]

        for lower_name, name in lowered.items():
            if re.search(rf"\b{re.escape(lower_name)}\b", cleaned):
                return name

        close = difflib.get_close_matches(cleaned, lowered.keys(), n=1, cutoff=0.8)
        if close:
            return lowered[close[0]]

        return None

    def _warn(self, message: str) -> None:
        print(f"  [{self.name}] {message}")
