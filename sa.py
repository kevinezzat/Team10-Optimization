"""Simulated Annealing for the existing Team10 UAV-UGV formulation.

Reuse `instance.Instance`, `fitness.decode`, and `fitness.evaluate` unchanged.
All solutions are encoded as the same 2*n random-key vectors used in Milestone 2.
The neighborhood is generated in route-space, then encoded back into keys.
"""
from dataclasses import dataclass
import math

import numpy as np

from fitness import (Result, combined_objective, decode, evaluate,
                     route_metrics, sa_fitness)
from instance import Instance, UGV_ONLY, UAV_ONLY


@dataclass(frozen=True)
class SAConfig:
    iterations: int = 12000
    initial_temperature: float = 0.08
    cooling_rate: float = 0.9995
    penalty: float = 100.0
    alpha: float = 0.5
    construction_restarts: int = 100


@dataclass
class SAResult:
    initial_x: np.ndarray
    initial_result: Result
    best_x: np.ndarray
    best_result: Result
    best_history: np.ndarray
    current_history: np.ndarray
    accepted: int
    attempted: int


def compatible(inst: Instance, k: int, task: int) -> bool:
    """Whether vehicle k may perform task id in 1..n."""
    task_type = inst.ttype[task - 1]
    return not ((task_type == UGV_ONLY and inst.is_uav(k)) or
                (task_type == UAV_ONLY and not inst.is_uav(k)))


def encode_routes(routes: list[list[int]], inst: Instance) -> np.ndarray:
    """Encode routes into existing random-key format without losing their order."""
    if len(routes) != inst.m:
        raise ValueError("expected one route per vehicle")
    task_ids = [task for route in routes for task in route]
    if sorted(task_ids) != list(range(1, inst.n + 1)):
        raise ValueError("each task 1..n must appear exactly once")
    x = np.empty(2 * inst.n, dtype=float)
    rank = 0
    for k, route in enumerate(routes):
        for task in route:
            x[task - 1] = (k + 0.5) / inst.m
            x[inst.n + task - 1] = (rank + 1) / (inst.n + 1)
            rank += 1
    return x


def _construct(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Greedy randomized, capability-aware insertion into vehicle routes.

    Hard resources are checked for each insertion. If no feasible insertion is
    possible, select the least-violating insertion and let SA repair it.
    """
    routes = [[] for _ in range(inst.m)]
    # Assign restricted tasks first, with mild randomized variation.
    tasks = list(range(1, inst.n + 1))
    rng.shuffle(tasks)
    tasks.sort(key=lambda i: inst.ttype[i - 1] == 0)

    metrics = [route_metrics(k, [], inst) for k in range(inst.m)]
    battery_total = float(inst.battery.sum())
    for task in tasks:
        choices = []
        for k in range(inst.m):
            if not compatible(inst, k, task):
                continue
            before = metrics[k]
            for pos in range(len(routes[k]) + 1):
                updated_route = routes[k][:pos] + [task] + routes[k][pos:]
                after = route_metrics(k, updated_route, inst)
                excess = (max(0.0, after['load'] - inst.payload[k]) / inst.payload[k]
                          + max(0.0, after['energy'] - inst.battery[k]) / inst.battery[k]
                          + max(0.0, after['time'] - inst.t_max) / inst.t_max)
                # Prefer feasible insertions first, then the least incremental
                # energy/time increase; jitter yields diverse SA restarts.
                delta = (0.5 * (after['time'] - before['time']) / inst.t_max
                         + 0.5 * (after['energy'] - before['energy']) / battery_total)
                score = (excess > 1e-10, excess, delta + rng.uniform(0.0, 0.002))
                choices.append((score, k, pos, updated_route, after))
        if not choices:
            raise ValueError(f"task {task} has no compatible vehicle")
        _, k, _, new_route, new_metrics = min(choices, key=lambda v: v[0])
        routes[k] = new_route
        metrics[k] = new_metrics
    return encode_routes(routes, inst)


def initial_solution(inst: Instance, rng: np.random.Generator,
                     restarts: int, alpha: float, penalty: float):
    """Find the best of several randomized greedy starts, preferring feasible."""
    if restarts < 1:
        raise ValueError("construction_restarts must be >= 1")
    best_x, best_result, best_score = None, None, float('inf')
    for _ in range(restarts):
        x = _construct(inst, rng)
        r = evaluate(x, inst)
        score = sa_fitness(r, inst, alpha, penalty)
        if score < best_score:
            best_x, best_result, best_score = x, r, score
    return best_x, best_result


def neighbor(x: np.ndarray, inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Move a task, swap tasks, or reverse part of a route.

    Capability is preserved (so g1 always remains 0). Payload, battery and
    deadlines are still checked by `evaluate`; SA penalizes violations.
    """
    old = decode(x, inst)
    for _ in range(40):
        routes = [r.copy() for r in old]
        operation = int(rng.integers(3))
        if operation == 0:  # relocate any task to another compatible vehicle
            sources = [k for k, route in enumerate(routes) if route]
            src = int(rng.choice(sources))
            src_pos = int(rng.integers(len(routes[src])))
            task = routes[src][src_pos]
            targets = [k for k in range(inst.m) if k != src and compatible(inst, k, task)]
            if not targets:
                continue
            dst = int(rng.choice(targets))
            routes[src].pop(src_pos)
            dst_pos = int(rng.integers(len(routes[dst]) + 1))
            routes[dst].insert(dst_pos, task)
        elif operation == 1:  # cross-vehicle exchange
            sources = [k for k, route in enumerate(routes) if route]
            if len(sources) < 2:
                continue
            a, b = rng.choice(sources, size=2, replace=False)
            a, b = int(a), int(b)
            ia, ib = int(rng.integers(len(routes[a]))), int(rng.integers(len(routes[b])))
            ta, tb = routes[a][ia], routes[b][ib]
            if not (compatible(inst, a, tb) and compatible(inst, b, ta)):
                continue
            routes[a][ia], routes[b][ib] = tb, ta
        else:  # 2-opt reversal within a route
            sources = [k for k, route in enumerate(routes) if len(route) >= 2]
            if not sources:
                continue
            k = int(rng.choice(sources))
            i, j = sorted(rng.choice(len(routes[k]), size=2, replace=False))
            routes[k][int(i):int(j) + 1] = routes[k][int(i):int(j) + 1][::-1]
        return encode_routes(routes, inst)
    return x.copy()  # no compatible modification available


def simulated_annealing(inst: Instance, config: SAConfig = SAConfig(),
                        seed: int = 1) -> SAResult:
    """Minimize the report's combined objective with a Metropolis SA search.

    The best feasible solution is tracked separately from the current state.
    Return a solution only if one satisfying all four constraints was found.
    """
    if not isinstance(config.iterations, int) or config.iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not (0 < config.cooling_rate < 1):
        raise ValueError("cooling_rate must lie strictly between 0 and 1")
    if not np.isfinite(config.initial_temperature) or config.initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive and finite")
    # Fail clearly instead of losing the makespan term when t_max is infinity.
    combined_objective(evaluate(np.full(2 * inst.n, 0.5), inst), inst, config.alpha)
    rng = np.random.default_rng(seed)
    current_x, current_r = initial_solution(inst, rng, config.construction_restarts,
                                            config.alpha, config.penalty)
    initial_x, initial_r = current_x.copy(), current_r
    current_score = sa_fitness(current_r, inst, config.alpha, config.penalty)
    best_x, best_r = (current_x.copy(), current_r) if current_r.violation <= 1e-10 else (None, None)
    best_cost = (combined_objective(best_r, inst, config.alpha)
                 if best_r is not None else float('inf'))

    best_hist = np.empty(config.iterations + 1)
    current_hist = np.empty(config.iterations + 1)
    best_hist[0], current_hist[0] = best_cost, current_score
    temperature, accepted = config.initial_temperature, 0
    for iteration in range(1, config.iterations + 1):
        candidate_x = neighbor(current_x, inst, rng)
        candidate_r = evaluate(candidate_x, inst)
        candidate_score = sa_fitness(candidate_r, inst, config.alpha, config.penalty)
        delta = candidate_score - current_score
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current_x, current_r, current_score = candidate_x, candidate_r, candidate_score
            accepted += 1
        if current_r.violation <= 1e-10:
            cost = combined_objective(current_r, inst, config.alpha)
            if cost < best_cost:
                best_x, best_r, best_cost = current_x.copy(), current_r, cost
        best_hist[iteration], current_hist[iteration] = best_cost, current_score
        temperature *= config.cooling_rate
    if best_r is None:
        raise RuntimeError("SA did not find a feasible solution: relax instance limits or increase restarts")
    return SAResult(initial_x, initial_r, best_x, best_r, best_hist,
                    current_hist, accepted, config.iterations)
