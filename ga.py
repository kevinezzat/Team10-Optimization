"""Hand-written random-key GA; no optimization toolbox is used."""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from fitness import evaluate, combined_objective
from sa import _construct, compatible
from sa_engine import ranking


@dataclass(frozen=True)
class GAConfig:
    population: int = 60
    evaluations: int = 6000
    crossover: float = .9
    mutation: float = .05
    elite: int = 2
    tournament: int = 3
    alpha: float = .5

    def __post_init__(self):
        for name in ('population', 'evaluations', 'elite', 'tournament'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.population < 2 or self.evaluations < self.population or self.elite >= self.population:
            raise ValueError('invalid population, evaluation budget or elite count')
        if any(not 0 <= v <= 1 for v in (self.crossover, self.mutation, self.alpha)):
            raise ValueError('probabilities and alpha must be in [0,1]')


def repair(x, inst, rng):
    """Repair capability only; resource excess remains visible to selection."""
    x = x.copy()
    for i in range(inst.n):
        k = min(int(x[i]*inst.m), inst.m-1)
        if not compatible(inst, k, i+1):
            allowed = [v for v in range(inst.m) if compatible(inst, v, i+1)]
            if not allowed:
                raise ValueError('task has no compatible vehicle')
            x[i] = (int(rng.choice(allowed)) + rng.random())/inst.m
    return x


def genetic_algorithm(inst, config=GAConfig(), seed=1):
    """Uniform crossover, per-gene reset mutation, tournaments and elitism.

    Every evaluated chromosome, including initialization, counts toward budget.
    Feasible solutions outrank infeasible ones; the latter rank by violation.
    """
    start = perf_counter()
    rng = np.random.default_rng(seed)
    combined_objective(evaluate(np.full(2*inst.n, .5), inst), inst, config.alpha)
    population = []
    history = []
    best = None
    count = 0

    def score(x):
        nonlocal best, count
        r = evaluate(x, inst)
        item = (ranking(r, inst, config.alpha), x.copy(), r)
        count += 1
        if best is None or item[0] < best[0]:
            best = item
        history.append((count, combined_objective(best[2], inst, config.alpha)
                        if best[2].violation <= 1e-9 else None))
        return item

    for i in range(config.population):
        x = (_construct(inst, rng, config.alpha) if i < config.population//2
             else repair(rng.random(2*inst.n), inst, rng))
        population.append(score(x))
    while count < config.evaluations:
        population.sort(key=lambda p: p[0])
        children = population[:config.elite]
        def parent():
            return min((population[int(rng.integers(len(population)))]
                        for _ in range(config.tournament)), key=lambda p:p[0])[1]
        while len(children) < config.population and count < config.evaluations:
            a, b = parent(), parent()
            x = np.where(rng.random(2*inst.n) < .5, a, b) if rng.random() < config.crossover else a.copy()
            mask = rng.random(2*inst.n) < config.mutation
            x[mask] = rng.random(int(mask.sum()))
            children.append(score(repair(x, inst, rng)))
        population = children
    return dict(best_x=best[1], best_result=best[2], history=history,
                evaluations=count, wall_time=perf_counter()-start,
                config=asdict(config), seed=seed)
