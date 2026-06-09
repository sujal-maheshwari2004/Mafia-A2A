"""A fully rule-based agent -- no LLM required.

Reasons from everything the A2A protocol exposes: its own role, private
knowledge (investigations, mafia teammates), and the full shape + content of
its personal feed.  Six kinds of signal are tracked and acted on each turn:

  accusation pressure   -- who is being named in suspicious/accusatory messages
  defense strength      -- who is being vouched for and trusted
  role claims           -- who has verbally claimed detective or doctor
  activity level        -- who is speaking and who has gone quiet
  private channel shape -- who is huddling privately with whom
  threat level          -- who is most actively accusing / investigating (Mafia view)

Information-seeking is a first-class behavior: the agent regularly asks direct
questions to put players on record, probes unverified role claims, and uses
silence as a signal worth pressuring.  A player who never asks questions is
half-blind -- they can only reason from what others volunteer.
"""

from __future__ import annotations

import random
from collections import Counter

from ..protocol import CastType, CommRequest
from ..models import Faction, Phase, Role
from .base import Agent, AgentView

_ACCUSATION_KEYWORDS = (
    "suspicious", "suspect", "mafia", "accuse", "vote out", "voting out",
    "lying", "liar", "hiding", "not who they say", "cover",
)
_DEFENSE_KEYWORDS = (
    "trust", "innocent", "clear", "vouch", "defend", "they're fine",
    "not mafia", "i believe", "back them",
)
_CLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "detective": ("detective", "investigated", "i investigated", "i checked", "came back"),
    "doctor": ("doctor", "i saved", "i protected", "i healed", "saved them"),
}

_PROBE_TEMPLATES = [
    "What's your actual read right now -- who are you watching?",
    "You've been quiet. Who are you suspicious of and why?",
    "Walk me through your thinking -- who do you think it is?",
    "I keep coming back to something. What's your gut telling you?",
    "Give me something concrete -- who are you actually looking at?",
    "You haven't said much. What have you been picking up on?",
    "What's your read on the people who've been loudest today?",
    "Is there anyone you're watching closely but haven't named yet?",
    "What do you make of everything you've heard so far?",
    "Who would you vote for right now if the vote was called this second?",
    "Anything in the last few exchanges stand out to you as off?",
    "Who's been the most believable person at this table today?",
]
_CHALLENGE_DETECTIVE = [
    "If you're really the detective, what exactly did you find?",
    "Okay, detective claim -- who did you investigate, and what came back?",
    "That's a bold claim. Give us the specifics.",
    "Alright, detective -- who did you check last night and what was the result?",
]
_CHALLENGE_DOCTOR = [
    "If you're the doctor, who did you protect last night?",
    "Doctor claim -- okay, but can you tell us who you've been saving?",
    "So who have you been protecting? I'd like the details.",
    "Alright, if you're the doctor, what's your read on who needs protection tonight?",
]
_MAFIA_COORD_TEMPLATES = [
    "Let's {verb} {target} -- they're our biggest threat right now.",
    "{target} is who I'm most worried about -- we should {verb} them.",
    "My read: {target} needs to go. They're getting too close.",
    "I say we {verb} {target} -- they've been asking the right questions.",
    "We need to {verb} {target} before they put it all together.",
    "I keep coming back to {target} -- {verb} them and we buy some time.",
]
_COVER_TEMPLATES = [
    "I actually don't buy that {name} is mafia -- this feels like a rush to judgment.",
    "Hold on. Everyone's piling on {name} but I haven't heard actual evidence.",
    "I've been watching {name} and honestly I'm not seeing it. What's the actual case here?",
    "Can we slow down on {name}? This feels like someone steering the room.",
]
_FRAME_TEMPLATES = [
    "Something feels off about {name} -- too quiet, too convenient.",
    "I keep coming back to {name}. What's everyone else's read?",
    "Can {name} actually explain their reasoning? Their story keeps shifting.",
    "{name} has been deflecting every time this comes up. That's a tell for me.",
]


class HeuristicAgent(Agent):
    def __init__(self, name: str, rng: random.Random | None = None, chattiness: float = 0.6):
        super().__init__(name)
        self._rng = rng or random.Random()
        self._chattiness = chattiness
        self._claimed = False
        # Keyed by "day:phase" -- tracks full normalised content of own sent messages
        self._said_fingerprints: dict[str, set[str]] = {}
        # Keyed by "day:phase" -- tracks players this agent has already probed
        self._probed_players: dict[str, set[str]] = {}
        # Keyed by "day:phase" -- tracks targets already suggested in mafia coordination
        self._coord_targets: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Night actions
    # ------------------------------------------------------------------
    def choose_night_target(self, view: AgentView) -> str:
        others = list(view.others_alive)

        if view.role is Role.MAFIA:
            pool = [n for n in others if n not in view.teammates]
            # Anyone who has publicly claimed detective is the single biggest threat
            claims = self._claim_tracker(view)
            detective_targets = [n for n in pool if claims.get(n) == "detective"]
            if detective_targets:
                return self._rng.choice(detective_targets)
            # Follow teammate consensus from the night huddle
            consensus = self._teammate_kill_suggestions(view, pool)
            if consensus:
                top = max(consensus.values())
                return self._rng.choice([n for n, c in consensus.items() if c == top])
            # Otherwise kill whoever has been most actively accusing / investigating
            threat = self._threat_tally(view, pool)
            if threat:
                top = max(threat.values())
                return self._rng.choice([n for n, c in threat.items() if c == top])
            return self._rng.choice(pool or others)

        if view.role is Role.DETECTIVE:
            unknown = [n for n in others if n not in view.known_factions]
            pool = unknown or others
            # Investigate whoever is under the most accusation pressure -- best chance of a hit
            pressure = self._accusation_pressure(view)
            ranked = sorted(pool, key=lambda n: pressure.get(n, 0), reverse=True)
            top_score = pressure.get(ranked[0], 0)
            return self._rng.choice([n for n in ranked if pressure.get(n, 0) == top_score])

        if view.role is Role.DOCTOR:
            # Protect whoever the Mafia is most likely targeting tonight
            claims = self._claim_tracker(view)
            # A detective claimer is the highest-value Mafia target -- protect them
            detective_claimers = [n for n in others if claims.get(n) == "detective"]
            if detective_claimers:
                return self._rng.choice(detective_claimers)
            # Next best: protect whoever has been most vocally accusing people
            # (they're a threat to Mafia, so Mafia wants them gone)
            activity = self._accusation_activity(view, others)
            if activity:
                top = max(activity.values())
                return self._rng.choice([n for n, c in activity.items() if c == top])
            # Fall back to self if under heat, otherwise random
            self_heat = self._accusation_pressure(view).get(view.self_name, 0)
            if self_heat > 0 and self._rng.random() < 0.6:
                return view.self_name
            return self._rng.choice(others) if others else view.self_name

        raise AssertionError(f"{view.role} has no night action")

    def _teammate_kill_suggestions(self, view: AgentView, candidates: list[str]) -> Counter[str]:
        """Names floated by teammates during tonight's huddle."""
        mentions: Counter[str] = Counter()
        candidate_set = set(candidates)
        for sighting in view.feed:
            if (
                sighting.sender in view.teammates
                and sighting.is_content_known
                and sighting.day_number == view.day_number
                and sighting.phase_label == Phase.NIGHT.value
            ):
                text = sighting.content.lower()
                for name in candidate_set:
                    if name.lower() in text:
                        mentions[name] += 1
        return mentions

    # ------------------------------------------------------------------
    # Discussion
    # ------------------------------------------------------------------
    def discussion_turn(self, view: AgentView) -> CommRequest | None:
        others = list(view.others_alive)
        if not others:
            return None

        phase_key = f"{view.day_number}:{view.phase.value}"

        own_said = sum(
            1 for s in view.feed
            if s.day_number == view.day_number
            and s.phase_label == view.phase.value
            and s.sender == view.self_name
        )
        total_said = sum(
            1 for s in view.feed
            if s.day_number == view.day_number
            and s.phase_label == view.phase.value
        )
        # Own messages drive personal decay heavily; total messages add a light shared dampening
        # so agents who've contributed a lot back off while silent observers also slow down gently.
        said_so_far = own_said * 3 + total_said * 0.1
        if self._rng.random() > self._chattiness / (1 + said_so_far * 0.15):
            return None

        # Detective with proof -> broadcast claim immediately
        if view.phase is Phase.DAY and view.role is Role.DETECTIVE and not self._claimed:
            mafia_found = [n for n in others if view.known_factions.get(n) is Faction.MAFIA]
            if mafia_found:
                self._claimed = True
                target = mafia_found[0]
                content = (
                    f"I'm the Detective. I investigated {target} last night and they came back "
                    f"Mafia -- we need to vote them out today."
                )
                return CommRequest(CastType.BROADCAST, (), content)

        if view.faction is Faction.MAFIA:
            result = self._mafia_turn(view, others, phase_key)
        else:
            result = self._town_turn(view, others, phase_key)

        if result is None:
            return None

        # Skip if this agent already said something identical this phase
        fingerprint = " ".join(result.content.lower().split())
        said = self._said_fingerprints.setdefault(phase_key, set())
        if fingerprint in said:
            return None
        said.add(fingerprint)
        return result

    def _mafia_turn(self, view: AgentView, others: list[str], phase_key: str) -> CommRequest | None:
        teammates_alive = [n for n in view.teammates if n in view.alive]
        non_mafia = [n for n in others if n not in view.teammates]

        # Priority 1 (day only): Cover a teammate drawing heat -- looks town, protects the team
        if view.phase is Phase.DAY:
            pressure = self._accusation_pressure(view)
            threatened = [n for n in teammates_alive if pressure.get(n, 0) >= 2]
            if threatened and self._rng.random() < 0.65:
                target = self._rng.choice(threatened)
                if teammates_alive and self._rng.random() < 0.45:
                    # Warn teammates privately first so they know to adjust
                    cast = CastType.UNICAST if len(teammates_alive) == 1 else CastType.MULTICAST
                    content = f"{target} is getting too much heat -- we need to redirect this somewhere else."
                    return CommRequest(cast, tuple(teammates_alive), content)
                # Publicly defend to blend in as a concerned townie
                content = self._rng.choice(_COVER_TEMPLATES).format(name=target)
                return CommRequest(CastType.BROADCAST, (), content)

        # Priority 2: Coordinate with teammates -- align on kill target or cover story
        # Skip targets already suggested this phase to avoid repeating the same message
        if teammates_alive and self._rng.random() < 0.4:
            coord_done = self._coord_targets.setdefault(phase_key, set())
            pool = [n for n in (non_mafia or others) if n not in coord_done]
            if pool:
                target = self._rng.choice(pool)
                coord_done.add(target)
                cast = CastType.UNICAST if len(teammates_alive) == 1 else CastType.MULTICAST
                verb = "eliminate" if view.phase is Phase.NIGHT else "vote out"
                content = self._rng.choice(_MAFIA_COORD_TEMPLATES).format(target=target, verb=verb)
                return CommRequest(cast, tuple(teammates_alive), content)

        # Priority 3 (day only): Frame a townie with a question that sounds like town concern
        if view.phase is Phase.DAY and non_mafia and self._rng.random() < 0.5:
            target = self._rng.choice(non_mafia)
            content = self._rng.choice(_FRAME_TEMPLATES).format(name=target)
            return CommRequest(CastType.BROADCAST, (), content)

        return None

    def _town_turn(self, view: AgentView, others: list[str], phase_key: str) -> CommRequest | None:
        pressure = self._accusation_pressure(view)
        claims = self._claim_tracker(view)
        probed = self._probed_players.setdefault(phase_key, set())

        # Priority 1: Challenge a role claim from someone already under suspicion
        # -- an unverified claim from a suspicious player is a tell worth pressing
        suspicious_claimers = [
            n for n, role_str in claims.items()
            if n in others and pressure.get(n, 0) > 0
        ]
        if suspicious_claimers and self._rng.random() < 0.55:
            target = self._rng.choice(suspicious_claimers)
            role_str = claims[target]
            templates = _CHALLENGE_DETECTIVE if role_str == "detective" else _CHALLENGE_DOCTOR
            question = self._rng.choice(templates)
            return CommRequest(CastType.BROADCAST, (), f"{target} -- {question}")

        # Priority 2: Strong confidence on a suspect -> accuse publicly or build a coalition first
        hard_suspects = [n for n in others if pressure.get(n, 0) >= 2]
        if hard_suspects and self._rng.random() < 0.5:
            target = self._rng.choice(hard_suspects)
            if self._rng.random() < 0.45:
                # Whisper to a confidant before going public -- test whether they agree
                pool = [n for n in others if n != target]
                if pool:
                    confidant = self._rng.choice(pool)
                    content = f"I'm pretty sure {target} is mafia. Are you seeing what I'm seeing?"
                    return CommRequest(CastType.UNICAST, (confidant,), content)
            content = f"I'm not letting this drop -- {target}'s story doesn't hold together and I'm voting for them."
            return CommRequest(CastType.BROADCAST, (), content)

        # Priority 3: Probe a player who hasn't spoken at all this phase
        # -- silence in Mafia is a choice, and it's worth naming out loud
        # Skip players already probed this phase so the same person isn't pressured repeatedly
        quiet = self._quiet_players(view, others)
        quiet_unprobed = [n for n in quiet if n not in probed]
        if quiet_unprobed and self._rng.random() < 0.45:
            target = self._rng.choice(quiet_unprobed)
            probed.add(target)
            question = self._rng.choice(_PROBE_TEMPLATES)
            return CommRequest(CastType.UNICAST, (target,), f"{target}, {question}")

        # Priority 4: Follow up on any unverified role claim -- even a non-suspicious one
        # deserves a question so we have something on the record
        unverified = [n for n in claims if n in others]
        if unverified and self._rng.random() < 0.4:
            target = self._rng.choice(unverified)
            role_str = claims[target]
            templates = _CHALLENGE_DETECTIVE if role_str == "detective" else _CHALLENGE_DOCTOR
            question = self._rng.choice(templates)
            return CommRequest(CastType.BROADCAST, (), f"Alright, {target} -- {question}")

        # Priority 5: Moderate suspicion -> broadcast accusation
        moderate_suspects = [n for n in others if pressure.get(n, 0) > 0]
        if moderate_suspects and self._rng.random() < 0.5:
            target = self._rng.choice(moderate_suspects)
            content = f"I think {target} is acting suspicious -- their story doesn't add up to me."
            return CommRequest(CastType.BROADCAST, (), content)

        # Priority 6: Direct probe to gather information -- skip already-probed targets
        unprobed_others = [n for n in others if n not in probed]
        if unprobed_others and self._rng.random() < 0.45:
            target = self._rng.choice(unprobed_others)
            probed.add(target)
            question = self._rng.choice(_PROBE_TEMPLATES)
            return CommRequest(CastType.UNICAST, (target,), f"Hey -- {question}")

        # Fallback: open question to the room
        target = self._rng.choice(others)
        content = f"I don't have a strong read yet. What does everyone think about {target}?"
        return CommRequest(CastType.BROADCAST, (), content)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------
    def choose_vote(self, view: AgentView) -> str | None:
        others = list(view.others_alive)
        if not others:
            return None

        if view.faction is Faction.TOWN:
            mafia_known = [n for n in others if view.known_factions.get(n) is Faction.MAFIA]
            if mafia_known:
                return self._rng.choice(mafia_known)

        pressure = self._accusation_pressure(view)
        pool = [n for n in others if n not in view.teammates] if view.faction is Faction.MAFIA else others

        ranked = sorted(pool, key=lambda n: pressure.get(n, 0), reverse=True)
        top_score = pressure.get(ranked[0], 0)
        if top_score > 0:
            return self._rng.choice([n for n in ranked if pressure.get(n, 0) == top_score])
        return self._rng.choice(pool)

    # ------------------------------------------------------------------
    # Feed analysis
    # ------------------------------------------------------------------
    def _accusation_pressure(self, view: AgentView) -> Counter[str]:
        """How many accusatory messages have named each player.

        Messages from the current day's discussion phase count 3× -- they
        represent fresh consensus that should dominate over older residue.
        """
        tally: Counter[str] = Counter()
        names = set(view.alive) - {view.self_name}
        for sighting in view.feed:
            if not sighting.is_content_known or sighting.sender == view.self_name:
                continue
            text = sighting.content.lower()
            if not any(kw in text for kw in _ACCUSATION_KEYWORDS):
                continue
            is_today_day = (
                sighting.day_number == view.day_number
                and sighting.phase_label == Phase.DAY.value
            )
            weight = 3 if is_today_day else 1
            for name in names:
                if name != sighting.sender and name.lower() in text:
                    tally[name] += weight
        return tally

    def _claim_tracker(self, view: AgentView) -> dict[str, str]:
        """First detected role claim per player (detective or doctor), from the feed."""
        claims: dict[str, str] = {}
        for sighting in view.feed:
            if not sighting.is_content_known or sighting.sender == view.self_name:
                continue
            if sighting.sender in claims:
                continue
            text = sighting.content.lower()
            for role_str, patterns in _CLAIM_PATTERNS.items():
                if any(p in text for p in patterns):
                    claims[sighting.sender] = role_str
                    break
        return claims

    def _threat_tally(self, view: AgentView, pool: list[str]) -> Counter[str]:
        """How actively each player in pool has been accusing or investigating (Mafia kill priority)."""
        tally: Counter[str] = Counter()
        pool_set = set(pool)
        threat_kw = _ACCUSATION_KEYWORDS + ("investigate", "detective", "found", "checking")
        for sighting in view.feed:
            if sighting.sender not in pool_set or not sighting.is_content_known:
                continue
            if any(kw in sighting.content.lower() for kw in threat_kw):
                tally[sighting.sender] += 1
        return tally

    def _accusation_activity(self, view: AgentView, pool: list[str]) -> Counter[str]:
        """How many accusatory messages each player in pool has sent (Doctor protection signal)."""
        tally: Counter[str] = Counter()
        pool_set = set(pool)
        for sighting in view.feed:
            if sighting.sender not in pool_set or not sighting.is_content_known:
                continue
            if any(kw in sighting.content.lower() for kw in _ACCUSATION_KEYWORDS):
                tally[sighting.sender] += 1
        return tally

    def _quiet_players(self, view: AgentView, others: list[str]) -> list[str]:
        """Players who have not spoken at all this phase -- silence is itself a signal.

        Counts all sightings by shape, not only messages whose content is readable.
        A player who sent a private whisper you couldn't read has still chosen to act
        -- treating them as silent would cause the whole table to pile on with probes.
        """
        speakers = {
            s.sender for s in view.feed
            if s.day_number == view.day_number
            and s.phase_label == view.phase.value
        }
        return [n for n in others if n not in speakers]
