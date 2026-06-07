"""A fully rule-based agent -- no LLM required.

It reasons only from what the A2A protocol actually exposes to it: its own
role, anything it has privately learned (investigations, mafia teammates),
and the shape + content of everything that has passed through its feed.
"Suspicion" is a live tally recomputed from the feed each time a decision is
needed -- there's no hidden state beyond what a human re-scanning the table
talk could plausibly notice. This keeps it a fair, swappable baseline next to
future LLM-backed agents that reason over the same view more richly.
"""

from __future__ import annotations

import random
from collections import Counter

from ..protocol import CastType, CommRequest
from ..models import Faction, Phase, Role
from .base import Agent, AgentView

_SUSPICION_KEYWORDS = ("suspicious", "suspect", "mafia", "accuse", "vote out", "lying", "liar")


class HeuristicAgent(Agent):
    def __init__(self, name: str, rng: random.Random | None = None, chattiness: float = 0.6):
        super().__init__(name)
        self._rng = rng or random.Random()
        self._chattiness = chattiness
        self._claimed = False

    # ------------------------------------------------------------------
    # Night actions
    # ------------------------------------------------------------------
    def choose_night_target(self, view: AgentView) -> str:
        others = list(view.others_alive)

        if view.role is Role.MAFIA:
            pool = [n for n in others if n not in view.teammates]
            consensus = self._teammate_kill_suggestions(view, pool)
            if consensus:
                top = max(consensus.values())
                return self._rng.choice([n for n, c in consensus.items() if c == top])
            return self._rng.choice(pool or others)

        if view.role is Role.DETECTIVE:
            unknown = [n for n in others if n not in view.known_factions]
            pool = unknown or others
            suspicion = self._suspicion_tally(view)
            ranked = sorted(pool, key=lambda n: suspicion.get(n, 0), reverse=True)
            top_score = suspicion.get(ranked[0], 0)
            return self._rng.choice([n for n in ranked if suspicion.get(n, 0) == top_score])

        if view.role is Role.DOCTOR:
            if others and self._rng.random() < 0.5:
                return self._rng.choice(others)
            return view.self_name

        raise AssertionError(f"{view.role} has no night action")

    def _teammate_kill_suggestions(self, view: AgentView, candidates: list[str]) -> Counter[str]:
        """What did my fellow mafia float during tonight's huddle?"""
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

        said_so_far = sum(
            1 for s in view.feed if s.day_number == view.day_number and s.phase_label == view.phase.value
        )
        if self._rng.random() > self._chattiness / (1 + said_so_far * 0.15):
            return None

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
            return self._mafia_turn(view, others)
        return self._town_turn(view, others)

    def _mafia_turn(self, view: AgentView, others: list[str]) -> CommRequest | None:
        teammates_alive = [n for n in view.teammates if n in view.alive]
        non_mafia = [n for n in others if n not in view.teammates]

        if teammates_alive and self._rng.random() < 0.4:
            target = self._rng.choice(non_mafia) if non_mafia else self._rng.choice(others)
            cast = CastType.UNICAST if len(teammates_alive) == 1 else CastType.MULTICAST
            verb = "kill" if view.phase is Phase.NIGHT else "vote out"
            content = f"Let's {verb} {target} -- they're the bigger threat to us."
            return CommRequest(cast, tuple(teammates_alive), content)

        if view.phase is Phase.DAY and non_mafia:
            target = self._rng.choice(non_mafia)
            content = f"Something feels off about {target} -- too quiet, too convenient. I'm suspicious of them."
            return CommRequest(CastType.BROADCAST, (), content)
        return None

    def _town_turn(self, view: AgentView, others: list[str]) -> CommRequest | None:
        suspicion = self._suspicion_tally(view)
        ranked = sorted((n for n in others if suspicion.get(n, 0) > 0), key=lambda n: suspicion[n], reverse=True)

        if ranked and self._rng.random() < 0.5:
            target = ranked[0]
            content = f"I think {target} is acting suspicious -- their story doesn't add up to me."
            return CommRequest(CastType.BROADCAST, (), content)

        if len(others) >= 2 and self._rng.random() < 0.35:
            confidant = self._rng.choice(others)
            remaining = [n for n in others if n != confidant]
            target = self._rng.choice(remaining or others)
            content = f"Just between us -- I don't trust {target}. Keep an eye on them."
            return CommRequest(CastType.UNICAST, (confidant,), content)

        target = self._rng.choice(others)
        content = f"I don't have a strong read yet, but {target} has been quiet. What do you all think?"
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

        suspicion = self._suspicion_tally(view)
        if view.faction is Faction.MAFIA:
            town_targets = [n for n in others if n not in view.teammates]
            pool = town_targets or others
        else:
            pool = others

        ranked = sorted(pool, key=lambda n: suspicion.get(n, 0), reverse=True)
        top_score = suspicion.get(ranked[0], 0)
        if top_score > 0:
            return self._rng.choice([n for n in ranked if suspicion.get(n, 0) == top_score])
        return self._rng.choice(pool)

    # ------------------------------------------------------------------
    # Reading the room: a live re-scan of the feed, not persisted state
    # ------------------------------------------------------------------
    def _suspicion_tally(self, view: AgentView) -> Counter[str]:
        tally: Counter[str] = Counter()
        names = set(view.alive) - {view.self_name}
        for sighting in view.feed:
            if not sighting.is_content_known or sighting.sender == view.self_name:
                continue
            text = sighting.content.lower()
            if not any(keyword in text for keyword in _SUSPICION_KEYWORDS):
                continue
            for name in names:
                if name != sighting.sender and name.lower() in text:
                    tally[name] += 1
        return tally
