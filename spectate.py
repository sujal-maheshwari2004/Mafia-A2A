"""Watch a full agent-vs-agent game play out over the A2A protocol.

Usage:
    python spectate.py [num_players] [seed] [--brain heuristic|llm|mixed] [--model gpt-4.1-mini]

`heuristic` agents are free and need no setup. `llm` and `mixed` use
LangChain + OpenAI and require OPENAI_API_KEY in your environment -- the
agents are interchangeable because both implement the same `Agent` interface
over the same agent-agnostic A2A protocol.
"""

import argparse
import os
import random
import sys

from mafia_sim import (
    Agent,
    CastType,
    DayResolved,
    GameEnded,
    GameEvent,
    GameStarted,
    HeuristicAgent,
    LLMAgent,
    NightResolved,
    PhaseStarted,
    Simulation,
    TableTalk,
    VoteCast,
    stable_int_seed,
)

DEFAULT_NAMES = [
    "Avery", "Bailey", "Casey", "Drew", "Ellis",
    "Frankie", "Harper", "Jordan", "Kai", "Logan",
]


def build_agent(brain: str, name: str, seed: int, model: str) -> Agent:
    rng = random.Random(stable_int_seed(name, seed))
    if brain == "llm":
        return LLMAgent(name, model=model, rng=rng)
    if brain == "mixed":
        # alternate so you can directly compare LLM vs. heuristic play in one game
        return LLMAgent(name, model=model, rng=rng) if stable_int_seed(name) % 2 else HeuristicAgent(name, rng=rng)
    return HeuristicAgent(name, rng=rng)


def render_event(event: GameEvent) -> None:
    """Render one structured `GameEvent` to the console.

    This is the *only* place in the spectator path that knows how to turn the
    game's structured narration into English -- a React frontend would do the
    same job differently (animations, copy, localisation) from the identical
    event stream a WebSocket would forward it verbatim.
    """
    match event:
        case GameStarted():
            pass  # the roster is already printed up front, before the game starts

        case PhaseStarted(phase="Night", day_number=day, present_count=count):
            print(f"\n--- Night {day}  ({count} awake) ---")
        case PhaseStarted(phase="Day", day_number=day, present=present):
            print(f"\n--- Day {day}  (at the table: {', '.join(present)}) ---")

        case TableTalk(sender=sender, cast=cast, to=to, content=content, day_number=day, phase=phase, room_size=room_size):
            shorthand = CastType(cast).shorthand
            room_tag = f" [room of {room_size}]" if phase == "Night" else ""
            print(f"    [{phase} {day}]{room_tag} {sender} -> {shorthand}: {','.join(to)} :: {content}")

        case VoteCast(voter=voter, target=target, tally_so_far=tally):
            cast = f"votes for {target}" if target else "abstains"
            standings = ", ".join(f"{name}={count}" for name, count in sorted(tally.items(), key=lambda kv: -kv[1]))
            print(f"    {voter} {cast}.  [tally so far: {standings or '(none yet)'}]")

        case NightResolved(killed=killed, saved=saved):
            if killed:
                print(f"{killed} was found dead this morning.")
            elif saved:
                print("The doctor's patient was attacked but survived the night!")
            else:
                print("No one died last night.")

        case DayResolved(lynched=lynched, lynched_role=role, tied=tied):
            if lynched:
                print(f"The town votes to lynch {lynched}, who was a {role}.")
            elif tied:
                print("The vote ended in a tie -- no one is lynched today.")
            else:
                print("No votes were cast -- no one is lynched today.")

        case GameEnded(winner=winner, roles=roles):
            print("\n=== Final roles ===")
            for name, role in roles.items():
                print(f"  {name}: {role}")
            print(f"\n{winner} wins!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", nargs="?", type=int, default=7, help="number of players (5-10)")
    parser.add_argument("seed", nargs="?", type=int, default=None, help="RNG seed for a reproducible game")
    parser.add_argument("--brain", choices=["heuristic", "llm", "mixed"], default="heuristic")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model for llm/mixed agents")
    args = parser.parse_args()

    if args.brain in ("llm", "mixed") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set -- export it before running with --brain llm/mixed.")

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    names = DEFAULT_NAMES[: args.count]
    agents = [build_agent(args.brain, name, seed, args.model) for name in names]

    kinds = ", ".join(f"{a.name}={type(a).__name__}" for a in agents)
    print(f"Seating {args.count} agents (seed={seed}, brain={args.brain}):\n  {kinds}\n")

    sim = Simulation(agents, rng_seed=seed)
    for event in sim.run():
        render_event(event)


if __name__ == "__main__":
    main()
