from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path

from config import GameConfig
from entities import Action


ACTION_SPACE: tuple[Action, ...] = tuple(Action)
POLICY_VERSION = 1


def observation_size_for_config(config: GameConfig) -> int:
    return 6 + config.enemy_count * 5


@dataclass
class LinearPolicyGenome:
    input_size: int
    output_size: int
    genes: list[float]

    @classmethod
    def random(cls, input_size: int, rng: random.Random, output_size: int | None = None, scale: float = 1.0) -> "LinearPolicyGenome":
        actual_output_size = len(ACTION_SPACE) if output_size is None else output_size
        gene_count = (input_size + 1) * actual_output_size
        genes = [rng.uniform(-scale, scale) for _ in range(gene_count)]
        return cls(input_size=input_size, output_size=actual_output_size, genes=genes)

    def clone(self) -> "LinearPolicyGenome":
        return LinearPolicyGenome(self.input_size, self.output_size, self.genes.copy())

    def act(self, observation: list[float]) -> Action:
        if len(observation) != self.input_size:
            raise ValueError(f"Expected observation size {self.input_size}, got {len(observation)}")

        stride = self.input_size + 1
        best_action_index = 0
        best_score = -math.inf

        for action_index in range(self.output_size):
            offset = action_index * stride
            score = self.genes[offset + self.input_size]
            for feature_index, value in enumerate(observation):
                score += self.genes[offset + feature_index] * value
            if score > best_score:
                best_score = score
                best_action_index = action_index

        return ACTION_SPACE[best_action_index]

    def crossover(self, other: "LinearPolicyGenome", rng: random.Random) -> "LinearPolicyGenome":
        if self.input_size != other.input_size or self.output_size != other.output_size:
            raise ValueError("Cannot crossover genomes with different shapes")

        genes = [self_gene if rng.random() < 0.5 else other_gene for self_gene, other_gene in zip(self.genes, other.genes)]
        return LinearPolicyGenome(self.input_size, self.output_size, genes)

    def mutate(self, rng: random.Random, mutation_rate: float, mutation_scale: float) -> None:
        for index, value in enumerate(self.genes):
            if rng.random() < mutation_rate:
                self.genes[index] = value + rng.gauss(0.0, mutation_scale)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": "linear",
            "version": POLICY_VERSION,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "action_space": [action.name for action in ACTION_SPACE],
            "genes": self.genes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LinearPolicyGenome":
        input_size = int(payload["input_size"])
        output_size = int(payload["output_size"])
        genes = [float(value) for value in payload["genes"]]
        expected_gene_count = (input_size + 1) * output_size
        if len(genes) != expected_gene_count:
            raise ValueError(f"Expected {expected_gene_count} genes, got {len(genes)}")
        return cls(input_size=input_size, output_size=output_size, genes=genes)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path: str | Path, expected_input_size: int | None = None) -> "LinearPolicyGenome":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        genome = cls.from_dict(payload)
        if expected_input_size is not None and genome.input_size != expected_input_size:
            raise ValueError(
                f"Genome expects observation size {genome.input_size}, but current config produces {expected_input_size}"
            )
        if genome.output_size != len(ACTION_SPACE):
            raise ValueError(f"Genome expects {genome.output_size} actions, but game exposes {len(ACTION_SPACE)}")
        return genome
