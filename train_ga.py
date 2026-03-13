from __future__ import annotations

from dataclasses import dataclass
import argparse
import random
from pathlib import Path

from config import GameConfig
from game import GameEnvironment
from genetic_ai import LinearPolicyGenome, observation_size_for_config


@dataclass
class EvaluationResult:
    fitness: float
    average_progress: float
    win_rate: float
    average_time: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a genetic player for the dodge runner game.")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--mutation-rate", type=float, default=0.08)
    parser.add_argument("--mutation-scale", type=float, default=0.35)
    parser.add_argument("--evaluation-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--output", type=Path, default=Path("best_genome.json"))
    parser.add_argument("--report-every", type=int, default=1)
    return parser


def evaluate_genome(
    genome: LinearPolicyGenome,
    config: GameConfig,
    seeds: list[int],
    max_steps: int,
) -> EvaluationResult:
    total_fitness = 0.0
    total_progress = 0.0
    total_wins = 0
    total_time = 0.0

    for seed in seeds:
        env = GameEnvironment(config=config, seed=seed)
        observation = env.reset(seed=seed)
        steps = 0

        while not env.is_done() and steps < max_steps:
            action = genome.act(observation)
            observation, _, _, _ = env.step(action)
            steps += 1

        info = env.get_info()
        total_fitness += float(info["fitness"])
        total_progress += float(info["progress_ratio"])
        total_time += float(info["elapsed_time"])
        if info["done_reason"] == "goal":
            total_wins += 1

    count = len(seeds)
    return EvaluationResult(
        fitness=total_fitness / count,
        average_progress=total_progress / count,
        win_rate=total_wins / count,
        average_time=total_time / count,
    )


def tournament_select(
    ranked_population: list[tuple[LinearPolicyGenome, EvaluationResult]],
    rng: random.Random,
    tournament_size: int,
) -> LinearPolicyGenome:
    contenders = rng.sample(ranked_population, k=min(tournament_size, len(ranked_population)))
    winner, _ = max(contenders, key=lambda item: item[1].fitness)
    return winner


def main() -> None:
    args = build_parser().parse_args()
    if args.population < 2:
        raise ValueError("Population must be at least 2")
    if args.elite_count < 1 or args.elite_count >= args.population:
        raise ValueError("Elite count must be between 1 and population - 1")
    if args.evaluation_seeds < 1:
        raise ValueError("evaluation-seeds must be at least 1")

    config = GameConfig()
    input_size = observation_size_for_config(config)
    rng = random.Random(args.base_seed)
    evaluation_seeds = [args.base_seed + offset for offset in range(args.evaluation_seeds)]

    population = [LinearPolicyGenome.random(input_size=input_size, rng=rng, scale=1.25) for _ in range(args.population)]
    best_overall: tuple[LinearPolicyGenome, EvaluationResult] | None = None

    for generation in range(1, args.generations + 1):
        ranked_population = [
            (genome, evaluate_genome(genome, config, evaluation_seeds, args.max_steps))
            for genome in population
        ]
        ranked_population.sort(key=lambda item: item[1].fitness, reverse=True)

        generation_best = ranked_population[0]
        if best_overall is None or generation_best[1].fitness > best_overall[1].fitness:
            best_overall = (generation_best[0].clone(), generation_best[1])
            best_overall[0].save(args.output)

        if generation == 1 or generation % args.report_every == 0:
            fitness_values = [result.fitness for _, result in ranked_population]
            average_fitness = sum(fitness_values) / len(fitness_values)
            print(
                "generation={generation} best_fitness={best:.2f} avg_fitness={avg:.2f} "
                "best_progress={progress:.1%} win_rate={wins:.1%} avg_time={time:.2f}s".format(
                    generation=generation,
                    best=generation_best[1].fitness,
                    avg=average_fitness,
                    progress=generation_best[1].average_progress,
                    wins=generation_best[1].win_rate,
                    time=generation_best[1].average_time,
                )
            )

        if generation == args.generations:
            break

        next_population = [genome.clone() for genome, _ in ranked_population[: args.elite_count]]
        while len(next_population) < args.population:
            parent_a = tournament_select(ranked_population, rng, args.tournament_size)
            parent_b = tournament_select(ranked_population, rng, args.tournament_size)
            child = parent_a.crossover(parent_b, rng)
            child.mutate(rng, mutation_rate=args.mutation_rate, mutation_scale=args.mutation_scale)
            next_population.append(child)
        population = next_population

    assert best_overall is not None
    print(
        "saved={path} best_fitness={fitness:.2f} best_progress={progress:.1%} win_rate={wins:.1%}".format(
            path=args.output,
            fitness=best_overall[1].fitness,
            progress=best_overall[1].average_progress,
            wins=best_overall[1].win_rate,
        )
    )


if __name__ == "__main__":
    main()
