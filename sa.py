"""Simulated Annealing for the existing Team10 UAV-UGV formulation.

Reuse `instance.Instance`, `fitness.decode`, and `fitness.evaluate` unchanged.
All solutions are encoded as the same 2*n random-key vectors used in Milestone 2.
The neighborhood is generated in route-space, then encoded back into keys.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field

import numpy as np

from fitness import (Result, combined_objective, decode, evaluate,
                     route_metrics, sa_fitness)
from instance import Instance, UGV_ONLY, UAV_ONLY


@dataclass(frozen=True)
class SAConfig:
    """Validated immutable search settings; temperatures use penalized cost units."""
    iterations: int = 12000
    initial_temperature: float | None = None
    cooling_rate: float = 0.9995
    penalty: float = 100.0
    alpha: float = 0.5
    construction_restarts: int = 100
    final_temperature: float = 1e-5
    derive_cooling: bool = True
    calibration_samples: int = 200
    target_acceptance: float = 0.8
    operators: tuple[str, ...] = ('relocate', 'swap', 'two_opt', 'intra_swap', 'or_opt')
    adaptive_operators: bool = False
    adaptive_penalty: bool = False
    stagnation_window: int = 1000
    reheat: bool = False
    n_restarts: int = 1
    acceptance_window: int = 100
    hill_climbing: bool = False

    def __post_init__(self):
        """Reject invalid settings before starting any expensive computation."""
        for name in ('iterations', 'construction_restarts', 'calibration_samples',
                     'stagnation_window', 'n_restarts', 'acceptance_window'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('final_temperature', 'initial_temperature'):
            value = getattr(self, name)
            if value is None and name == 'initial_temperature':
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 < self.cooling_rate < 1 or not 0 < self.target_acceptance < 1:
            raise ValueError('cooling rate and target acceptance must be in (0, 1)')
        if not 0 <= self.alpha <= 1 or not np.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError('invalid alpha or penalty')
        allowed = {'relocate', 'swap', 'two_opt', 'intra_swap', 'or_opt'}
        if not self.operators or len(set(self.operators)) != len(self.operators) or not set(self.operators) <= allowed:
            raise ValueError('operators must be distinct supported names')

    def to_dict(self):
        """Return a JSON-serializable configuration."""
        return asdict(self)


@dataclass
class SAResult:
    """Search outcome, with complete histories for the selected independent chain."""
    initial_x: np.ndarray
    initial_result: Result
    best_x: np.ndarray
    best_result: Result
    best_history: np.ndarray
    current_history: np.ndarray
    accepted: int
    attempted: int
    temperature_history: np.ndarray = field(default_factory=lambda: np.array([]))
    acceptance_history: np.ndarray = field(default_factory=lambda: np.array([]))
    operator_stats: dict = field(default_factory=dict)
    evaluations: int = 0
    wall_time: float = 0.0
    feasible: bool = False
    config: dict = field(default_factory=dict)
    seed: int = 1
    best_chain: int = 0
    failed_moves: int = 0
    snapshots: list = field(default_factory=list)


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


def _construct(inst: Instance, rng: np.random.Generator, alpha: float = .5) -> np.ndarray:
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
            # Vectorized edge deltas score every insertion without rebuilding tours.
            seq = np.asarray([0]+routes[k]+[0])
            matrix = inst.dist(k)
            extra = matrix[seq[:-1], task]+matrix[task, seq[1:]]-matrix[seq[:-1], seq[1:]]
            times = before['time']+extra/inst.speed[k]+inst.service[task-1]
            energies = before['energy']+extra*inst.e_rate[k]
            load = before['load']+inst.demand[task-1]
            excess = (max(0., load-inst.payload[k])/inst.payload[k]
                      + np.maximum(0., energies-inst.battery[k])/inst.battery[k]
                      + np.maximum(0., times-inst.t_max)/inst.t_max)
            delta = (alpha*(times-before['time'])/inst.t_max
                     + (1-alpha)*(energies-before['energy'])/battery_total
                     + rng.uniform(0., .002, size=len(extra)))
            pos = int(np.lexsort((delta, excess, excess > 1e-10))[0])
            after = dict(dist=before['dist']+extra[pos], time=times[pos],
                         energy=energies[pos], load=load)
            score = (excess[pos] > 1e-10, excess[pos], delta[pos])
            updated_route = routes[k][:pos]+[task]+routes[k][pos:]
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
    best_x, best_result, best_score = None, None, (True, float('inf'), float('inf'))
    for _ in range(restarts):
        x = _construct(inst, rng, alpha)
        r = evaluate(x, inst)
        score = (r.violation > 1e-9, r.violation if r.violation > 1e-9 else 0., combined_objective(r, inst, alpha))
        if score < best_score:
            best_x, best_result, best_score = x, r, score
    return best_x, best_result


def neighbor(x: np.ndarray, inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Compatibility adapter for callers using random keys; SA uses route states."""
    from sa_engine import RouteState, propose
    state = RouteState(decode(x, inst), inst)
    for op in rng.permutation(SAConfig().operators):
        for _ in range(8):
            candidate = propose(state, inst, rng, op)
            if candidate is not None:
                return encode_routes(candidate.routes, inst)
    return x.copy()


def simulated_annealing(inst: Instance, config: SAConfig = SAConfig(),
                        seed: int = 1) -> SAResult:
    """Minimize the report objective; return the least violation if infeasible."""
    from sa_engine import run
    return run(inst, config, seed)
