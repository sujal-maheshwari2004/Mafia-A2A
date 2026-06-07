"""Game engine for the Mafia simulator.

This module contains no input()/print() calls -- it is pure game logic that
returns structured results and a narrative log. A CLI, web API, or test suite
can all drive a game through this same interface.
"""

from __future__ import annotations

import random
from collections import Counter

from .models import (
    DayResult,
    Faction,
    NightResult,
    Phase,
    Player,
    Role,
)


class IllegalActionError(Exception):
    """Raised when an action is attempted that the current game state forbids."""


class GameEngine:
    def __init__(self, player_names: list[str], num_mafia: int | None = None, rng_seed: int | None = None):
        if len(player_names) < 5:
            raise ValueError("Mafia needs at least 5 players to be playable")
        if len(set(player_names)) != len(player_names):
            raise ValueError("Player names must be unique")

        self._rng = random.Random(rng_seed)
        self._names = list(player_names)
        self._num_mafia = num_mafia if num_mafia is not None else max(1, len(player_names) // 4)
        if self._num_mafia >= len(player_names):
            raise ValueError("Too many mafia for the number of players")

        self.players: list[Player] = []
        self.phase: Phase = Phase.LOBBY
        self.day_number: int = 0
        self.log: list[str] = []
        self.winner: Faction | None = None

        self._night_actions: dict[str, str] = {}  # actor name -> target name
        self._day_votes: dict[str, str | None] = {}  # voter name -> target name (None = abstain)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.phase is not Phase.LOBBY:
            raise IllegalActionError("Game has already started")

        roles = self._build_role_list()
        self._rng.shuffle(roles)
        self.players = [Player(name, role) for name, role in zip(self._names, roles)]

        self.phase = Phase.NIGHT
        self.day_number = 1
        self._log(f"Night {self.day_number} falls. Everyone closes their eyes.")

    def _build_role_list(self) -> list[Role]:
        roles = [Role.MAFIA] * self._num_mafia
        remaining = len(self._names) - self._num_mafia

        for special in (Role.DOCTOR, Role.DETECTIVE):
            if remaining > 0:
                roles.append(special)
                remaining -= 1

        roles.extend([Role.VILLAGER] * remaining)
        return roles

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_player(self, name: str) -> Player:
        for player in self.players:
            if player.name == name:
                return player
        raise KeyError(f"No such player: {name}")

    @property
    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    @property
    def mafia_alive(self) -> list[Player]:
        return [p for p in self.alive_players if p.faction is Faction.MAFIA]

    @property
    def town_alive(self) -> list[Player]:
        return [p for p in self.alive_players if p.faction is Faction.TOWN]

    def players_with_role(self, role: Role, alive_only: bool = True) -> list[Player]:
        pool = self.alive_players if alive_only else self.players
        return [p for p in pool if p.role is role]

    # ------------------------------------------------------------------
    # Night phase
    # ------------------------------------------------------------------
    def submit_night_action(self, actor_name: str, target_name: str) -> None:
        if self.phase is not Phase.NIGHT:
            raise IllegalActionError("It is not night")

        actor = self.get_player(actor_name)
        target = self.get_player(target_name)

        if not actor.alive:
            raise IllegalActionError(f"{actor_name} is dead and cannot act")
        if not target.alive:
            raise IllegalActionError(f"{target_name} is dead and cannot be targeted")
        if actor.role not in (Role.MAFIA, Role.DOCTOR, Role.DETECTIVE):
            raise IllegalActionError(f"{actor_name} has no night action")
        if actor.role is Role.MAFIA and target.faction is Faction.MAFIA:
            raise IllegalActionError("Mafia cannot target their own teammates")

        self._night_actions[actor_name] = target_name

    def ready_to_resolve_night(self) -> bool:
        """True once every alive player with a night action has acted."""
        actors = [p for p in self.alive_players if p.role in (Role.MAFIA, Role.DOCTOR, Role.DETECTIVE)]
        return all(p.name in self._night_actions for p in actors)

    def resolve_night(self) -> NightResult:
        if self.phase is not Phase.NIGHT:
            raise IllegalActionError("It is not night")

        mafia_target = self._tally_mafia_kill()
        doctor_target = self._single_action_target(Role.DOCTOR)
        investigations = self._run_investigations()

        saved = mafia_target is not None and mafia_target == doctor_target
        killed: str | None = None
        if mafia_target is not None and not saved:
            victim = self.get_player(mafia_target)
            victim.alive = False
            killed = victim.name

        result = NightResult(
            day_number=self.day_number,
            killed=killed,
            saved=saved,
            investigations=investigations,
        )

        self._narrate_night(result)
        self._night_actions.clear()

        if not self._conclude_if_game_over():
            self.phase = Phase.DAY
            self._log(f"Day {self.day_number} begins. Discuss and vote to lynch a suspect.")

        return result

    def _tally_mafia_kill(self) -> str | None:
        mafia = self.players_with_role(Role.MAFIA)
        votes = [self._night_actions[m.name] for m in mafia if m.name in self._night_actions]
        if not votes:
            return None
        counts = Counter(votes)
        top = max(counts.values())
        winners = [name for name, count in counts.items() if count == top]
        return self._rng.choice(winners)

    def _single_action_target(self, role: Role) -> str | None:
        actors = self.players_with_role(role)
        if not actors:
            return None
        actor = actors[0]
        return self._night_actions.get(actor.name)

    def _run_investigations(self) -> dict[str, tuple[str, Faction]]:
        results: dict[str, tuple[str, Faction]] = {}
        for detective in self.players_with_role(Role.DETECTIVE):
            target_name = self._night_actions.get(detective.name)
            if target_name is not None:
                target = self.get_player(target_name)
                results[detective.name] = (target.name, target.faction)
        return results

    def _narrate_night(self, result: NightResult) -> None:
        if result.killed:
            self._log(f"{result.killed} was found dead this morning.")
        elif result.saved:
            self._log("The doctor's patient was attacked but survived the night!")
        else:
            self._log("No one died last night.")

    # ------------------------------------------------------------------
    # Day phase
    # ------------------------------------------------------------------
    def submit_vote(self, voter_name: str, target_name: str | None) -> None:
        if self.phase is not Phase.DAY:
            raise IllegalActionError("It is not day")

        voter = self.get_player(voter_name)
        if not voter.alive:
            raise IllegalActionError(f"{voter_name} is dead and cannot vote")

        if target_name is not None:
            target = self.get_player(target_name)
            if not target.alive:
                raise IllegalActionError(f"{target_name} is dead and cannot be voted for")

        self._day_votes[voter_name] = target_name

    def ready_to_resolve_day(self) -> bool:
        return all(p.name in self._day_votes for p in self.alive_players)

    def resolve_day(self) -> DayResult:
        if self.phase is not Phase.DAY:
            raise IllegalActionError("It is not day")

        cast_votes = [target for target in self._day_votes.values() if target is not None]
        counts = Counter(cast_votes)
        vote_counts = dict(counts)

        lynched: str | None = None
        tied = False
        if counts:
            top = max(counts.values())
            leaders = [name for name, count in counts.items() if count == top]
            if len(leaders) == 1:
                lynched = leaders[0]
                victim = self.get_player(lynched)
                victim.alive = False
            else:
                tied = True

        result = DayResult(
            day_number=self.day_number,
            lynched=lynched,
            vote_counts=vote_counts,
            tied=tied,
        )

        self._narrate_day(result)
        self._day_votes.clear()

        if not self._conclude_if_game_over():
            self.day_number += 1
            self.phase = Phase.NIGHT
            self._log(f"Night {self.day_number} falls. Everyone closes their eyes.")

        return result

    def _narrate_day(self, result: DayResult) -> None:
        if result.lynched:
            victim = self.get_player(result.lynched)
            self._log(f"The town votes to lynch {victim.name}, who was a {victim.role.value}.")
        elif result.tied:
            self._log("The vote ended in a tie -- no one is lynched today.")
        else:
            self._log("No votes were cast -- no one is lynched today.")

    # ------------------------------------------------------------------
    # Win conditions
    # ------------------------------------------------------------------
    def _conclude_if_game_over(self) -> bool:
        winner = self._check_winner()
        if winner is not None:
            self.winner = winner
            self.phase = Phase.ENDED
            self._log(f"Game over -- {winner.value} wins!")
            return True
        return False

    def _check_winner(self) -> Faction | None:
        mafia_count = len(self.mafia_alive)
        town_count = len(self.town_alive)

        if mafia_count == 0:
            return Faction.TOWN
        if mafia_count >= town_count:
            return Faction.MAFIA
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.log.append(message)
