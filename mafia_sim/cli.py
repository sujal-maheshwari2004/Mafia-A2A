"""Text-based driver for the Mafia simulator.

You play one seat at the table; the rest are filled by simple AI bots.
This module only handles input/output -- all game rules live in `engine.py`.
"""

from __future__ import annotations

import random

from .engine import GameEngine
from .models import Faction, Phase, Player, Role

BANNER = r"""
=======================================
            M A F I A
   a town full of secrets and lies
=======================================
"""

BOT_NAME_POOL = [
    "Avery", "Bailey", "Casey", "Drew", "Ellis", "Frankie",
    "Harper", "Jordan", "Kai", "Logan", "Morgan", "Parker",
    "Quinn", "Reese", "Sage", "Skyler",
]

ROLE_BLURBS = {
    Role.MAFIA: "Each night you and your fellow mafia choose a townsperson to eliminate. "
                "You win once the mafia equal or outnumber the town.",
    Role.DOCTOR: "Each night you may protect one player (including yourself) from the mafia's attack.",
    Role.DETECTIVE: "Each night you may investigate one player to learn whether "
                    "they belong to the Town or the Mafia.",
    Role.VILLAGER: "You have no special power -- only your voice and your vote. "
                   "Find the mafia before they find you.",
}


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run() -> None:
    print(BANNER)
    try:
        while True:
            play_one_game()
            if not prompt_yes_no("Play again? [y/N] "):
                print("Thanks for playing!")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


def play_one_game() -> None:
    total = ask_player_count()
    human_name = ask_human_name()
    names = build_player_names(human_name, total)

    engine = GameEngine(names)
    engine.start()
    last_log_idx = len(engine.log)  # the "night falls" line is shown via our own header

    print(f"\nTonight's table: {', '.join(names)}")
    reveal_role(engine, human_name)

    bot_memory: dict[str, dict[str, Faction]] = {}
    while engine.phase is not Phase.ENDED:
        if engine.phase is Phase.NIGHT:
            last_log_idx = run_night(engine, human_name, bot_memory, last_log_idx)
        elif engine.phase is Phase.DAY:
            last_log_idx = run_day(engine, human_name, bot_memory, last_log_idx)

    show_results(engine, human_name)


# ----------------------------------------------------------------------
# Setup prompts
# ----------------------------------------------------------------------
def ask_player_count() -> int:
    while True:
        raw = input("How many players at the table (5-12, default 6)? ").strip()
        if not raw:
            return 6
        if raw.isdigit() and 5 <= int(raw) <= 12:
            return int(raw)
        print("Please enter a number between 5 and 12.")


def ask_human_name() -> str:
    while True:
        name = input("What's your name? ").strip()
        if name:
            return name
        print("Please enter a non-empty name.")


def build_player_names(human_name: str, total: int) -> list[str]:
    pool = [n for n in BOT_NAME_POOL if n.lower() != human_name.lower()]
    bot_names = random.sample(pool, total - 1)
    names = [human_name] + bot_names
    random.shuffle(names)
    return names


def reveal_role(engine: GameEngine, human_name: str) -> None:
    player = engine.get_player(human_name)
    print("\n" + "-" * 50)
    print(f"{human_name}, your secret role is: {player.role.value}")
    print(ROLE_BLURBS[player.role])
    if player.role is Role.MAFIA:
        teammates = [p.name for p in engine.players if p.role is Role.MAFIA and p.name != human_name]
        if teammates:
            print("Your fellow mafia: " + ", ".join(teammates))
    print("-" * 50)
    input("Press Enter when you're ready to begin...")


# ----------------------------------------------------------------------
# Night phase
# ----------------------------------------------------------------------
def run_night(engine: GameEngine, human_name: str, bot_memory: dict, last_log_idx: int) -> int:
    print(f"\n--- Night {engine.day_number} ---")
    actors = [p for p in engine.alive_players if p.role in (Role.MAFIA, Role.DOCTOR, Role.DETECTIVE)]

    for player in actors:
        if player.name == human_name:
            target = human_night_choice(engine, player)
        else:
            target = bot_night_action(engine, player, bot_memory)
        engine.submit_night_action(player.name, target)

    result = engine.resolve_night()

    if human_name in result.investigations:
        target_name, faction = result.investigations[human_name]
        print(f"\n(Private) Your investigation reveals that {target_name} is aligned with the {faction.value}.")

    for detective_name, (target_name, faction) in result.investigations.items():
        if detective_name != human_name:
            bot_memory.setdefault(detective_name, {})[target_name] = faction

    return flush_log(engine, last_log_idx)


def human_night_choice(engine: GameEngine, player: Player) -> str:
    alive = engine.alive_players
    if player.role is Role.MAFIA:
        candidates = [p for p in alive if p.faction is not Faction.MAFIA]
        return prompt_target("Choose a target to eliminate tonight:", candidates)
    if player.role is Role.DOCTOR:
        return prompt_target("Choose someone to protect tonight (you may pick yourself):", alive)
    if player.role is Role.DETECTIVE:
        candidates = [p for p in alive if p.name != player.name]
        return prompt_target("Choose someone to investigate tonight:", candidates)
    raise AssertionError(f"{player.role} has no night action")


def bot_night_action(engine: GameEngine, player: Player, bot_memory: dict) -> str:
    alive = engine.alive_players

    if player.role is Role.MAFIA:
        candidates = [p for p in alive if p.faction is not Faction.MAFIA]
        return random.choice(candidates).name

    if player.role is Role.DOCTOR:
        return random.choice(alive).name

    if player.role is Role.DETECTIVE:
        memory = bot_memory.setdefault(player.name, {})
        unknown = [p for p in alive if p.name != player.name and p.name not in memory]
        pool = unknown or [p for p in alive if p.name != player.name]
        return random.choice(pool).name

    raise AssertionError(f"{player.role} has no night action")


# ----------------------------------------------------------------------
# Day phase
# ----------------------------------------------------------------------
def run_day(engine: GameEngine, human_name: str, bot_memory: dict, last_log_idx: int) -> int:
    print(f"\n--- Day {engine.day_number} ---")
    alive = engine.alive_players
    print("Still standing: " + ", ".join(p.name for p in alive))

    for player in alive:
        if player.name == human_name:
            candidates = [p for p in alive if p.name != player.name]
            target = prompt_target("Who do you vote to lynch?", candidates, allow_abstain=True)
        else:
            target = bot_day_vote(engine, player, bot_memory)
        engine.submit_vote(player.name, target)

    engine.resolve_day()
    return flush_log(engine, last_log_idx)


def bot_day_vote(engine: GameEngine, player: Player, bot_memory: dict) -> str | None:
    alive = [p for p in engine.alive_players if p.name != player.name]
    if not alive:
        return None

    if player.role is Role.DETECTIVE:
        memory = bot_memory.get(player.name, {})
        suspects = [p for p in alive if memory.get(p.name) is Faction.MAFIA]
        if suspects:
            return random.choice(suspects).name

    if player.faction is Faction.MAFIA:
        town_targets = [p for p in alive if p.faction is not Faction.MAFIA]
        if town_targets:
            return random.choice(town_targets).name

    return random.choice(alive).name


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def prompt_target(prompt_text: str, candidates: list[Player], allow_abstain: bool = False) -> str | None:
    print(prompt_text)
    for i, p in enumerate(candidates, start=1):
        print(f"  {i}. {p.name}")
    if allow_abstain:
        print(f"  {len(candidates) + 1}. Abstain")

    while True:
        choice = input("> ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1].name
            if allow_abstain and idx == len(candidates) + 1:
                return None
        print("Please enter a valid option number.")


def prompt_yes_no(prompt_text: str) -> bool:
    return input(prompt_text).strip().lower() in ("y", "yes")


def flush_log(engine: GameEngine, start_idx: int) -> int:
    for entry in engine.log[start_idx:]:
        print(entry)
    return len(engine.log)


def show_results(engine: GameEngine, human_name: str) -> None:
    print("\n=== Final roles ===")
    for p in engine.players:
        marker = "  <- you" if p.name == human_name else ""
        status = "alive" if p.alive else "dead"
        print(f"  {p.name}: {p.role.value} ({status}){marker}")
    print(f"\n{engine.winner.value} wins the game!")
