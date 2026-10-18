"""Cached route-state engine; public random-key adapters live in sa.py."""
from __future__ import annotations

import logging
import math
import time
from collections import deque
import numpy as np
from fitness import Result, combined_objective, decode, evaluate, route_metrics, sa_fitness


class RouteState:
    """Persistent route state: proposals replace only affected routes and metrics."""

    def __init__(self, routes, inst, metrics=None, capability=None):
        """Build a cache, or adopt a proposal's shallow-copied metric list."""
        self.routes = routes
        self.metrics = metrics if metrics is not None else [
            route_metrics(k, route, inst) for k, route in enumerate(routes)]
        self.capability = capability if capability is not None else [
            self.capability_violation(k, route, inst) for k, route in enumerate(routes)]
        self.result = self.aggregate(inst)

    @staticmethod
    def capability_violation(k, route, inst):
        """Cache the normalized capability count alongside each vehicle's metrics."""
        return sum((inst.ttype[t-1] == 1 and k >= inst.m_g) or
                   (inst.ttype[t-1] == 2 and k < inst.m_g) for t in route)/inst.n

    def aggregate(self, inst):
        """Rebuild report objectives and normalized constraints from cached metrics."""
        cap = sum(self.capability)
        payload = sum(max(0., r['load']-inst.payload[k])/inst.payload[k]
                      for k, r in enumerate(self.metrics))
        energy = sum(max(0., r['energy']-inst.battery[k])/inst.battery[k]
                     for k, r in enumerate(self.metrics))
        deadline = sum(max(0., r['time']-inst.t_max)/inst.t_max
                       for r in self.metrics) if np.isfinite(inst.t_max) else 0.
        details = dict(per_vehicle=self.metrics, g_capability=float(cap),
                       g_payload=payload, g_energy=energy, g_deadline=deadline)
        return Result(max(r['time'] for r in self.metrics),
                      sum(r['energy'] for r in self.metrics),
                      float(cap+payload+energy+deadline), self.routes, details)

    def changed(self, updates, inst):
        """Evaluate only one or two changed vehicles without mutating this state."""
        routes, metrics = self.routes.copy(), self.metrics.copy()
        capability = self.capability.copy()
        for k, route in updates.items():
            routes[k] = route
            metrics[k] = route_metrics(k, route, inst)
            capability[k] = self.capability_violation(k, route, inst)
        return RouteState(routes, inst, metrics, capability)


def propose(state, inst, rng, operator):
    """Generate a nonidentity compatible move; return None if this draw fails."""
    from sa import compatible
    routes = state.routes
    sources = [k for k, route in enumerate(routes) if route]
    if operator in ('two_opt', 'intra_swap'):
        sources = [k for k in sources if len(routes[k]) >= 2]
    if not sources:
        return None
    a = int(rng.choice(sources))
    ra = routes[a].copy()
    i = int(rng.integers(len(ra)))
    if operator in ('two_opt', 'intra_swap'):
        i, j = sorted(map(int, rng.choice(len(ra), 2, replace=False)))
        if operator == 'two_opt':
            ra[i:j+1] = reversed(ra[i:j+1])
        else:
            ra[i], ra[j] = ra[j], ra[i]
        return state.changed({a: ra}, inst)
    if operator == 'swap':
        choices = [(b, j) for b in sources if b != a
                   for j, task in enumerate(routes[b])
                   if compatible(inst, b, ra[i]) and compatible(inst, a, task)]
        if not choices:
            return None
        b, j = choices[int(rng.integers(len(choices)))]
        rb = routes[b].copy()
        ra[i], rb[j] = rb[j], ra[i]
        return state.changed({a: ra, b: rb}, inst)
    length = 1 if operator == 'relocate' else int(rng.integers(1, min(3, len(ra)-i)+1))
    segment = ra[i:i+length]
    targets = [b for b in range(inst.m)
               if (b != a or operator == 'or_opt')
               and all(compatible(inst, b, task) for task in segment)]
    if not targets:
        return None
    b = int(rng.choice(targets))
    del ra[i:i+length]
    rb = ra if b == a else routes[b].copy()
    j = int(rng.integers(len(rb)+1))
    if operator == 'or_opt' and rng.random() < .5:
        segment.reverse()
    rb[j:j] = segment
    updates = {a: ra, b: rb}
    if all(routes[k] == route for k, route in updates.items()):
        return None
    return state.changed(updates, inst)


def ranking(result, inst, alpha):
    """Feasible first, then objective; infeasible candidates by violation then F."""
    return (result.violation > 1e-9,
            result.violation if result.violation > 1e-9 else 0.,
            combined_objective(result, inst, alpha))


def run_chain(inst, config, rng, seed, chain):
    """Execute one annealing chain, retaining feasible and least-violating elites."""
    from sa import SAResult, encode_routes, initial_solution
    initial_x, initial = initial_solution(inst, rng, config.construction_restarts,
                                         config.alpha, config.penalty)
    current = RouteState(decode(initial_x, inst), inst)
    best = current
    penalty = config.penalty
    evaluations = config.construction_restarts
    samples = []
    # Calibration samples the actual penalized landscape, including uphill infeasibility.
    for _ in range(config.calibration_samples):
        candidate = propose(current, inst, rng, str(rng.choice(config.operators)))
        if candidate is not None:
            evaluations += 1
            samples.append(abs(sa_fitness(candidate.result, inst, config.alpha, penalty)
                               - sa_fitness(current.result, inst, config.alpha, penalty)))
    scale = float(np.mean(samples)) if samples else 0.
    t0 = config.initial_temperature or max(scale / -math.log(config.target_acceptance),
                                          config.final_temperature * 10)
    rate = ((config.final_temperature/t0)**(1/max(1, config.iterations-1))
            if config.derive_cooling else config.cooling_rate)
    temperature = t0
    stats = {op: dict(attempt=0, accept=0, improve=0, failed=0) for op in config.operators}
    recent = {op: deque(maxlen=100) for op in config.operators}
    outcomes = deque(maxlen=config.acceptance_window)
    best_hist, current_hist, temp_hist, acceptance_hist = [], [], [], []
    snapshots = []
    accepted = failed = stagnant = infeasible_steps = 0
    for step in range(config.iterations+1):
        best_hist.append(combined_objective(best.result, inst, config.alpha)
                         if best.result.violation <= 1e-9 else float('nan'))
        current_hist.append(sa_fitness(current.result, inst, config.alpha, penalty))
        temp_hist.append(0. if config.hill_climbing else temperature)
        acceptance_hist.append(float(np.mean(outcomes)) if outcomes else 0.)
        if step % max(1, config.iterations//4) == 0:
            snapshots.append((step, [r.copy() for r in best.routes]))
        if step == config.iterations:
            break
        candidate = None
        weights = np.array([.1 + (np.mean(recent[op]) if recent[op] else .5)
                            for op in config.operators])
        order = rng.choice(len(config.operators), len(config.operators), replace=False,
                           p=weights/weights.sum() if config.adaptive_operators else None)
        # Retry other neighborhoods rather than counting identity proposals as accepted.
        for index in order:
            op = config.operators[int(index)]
            for _ in range(8):
                stats[op]['attempt'] += 1
                candidate = propose(current, inst, rng, op)
                if candidate is not None:
                    break
                stats[op]['failed'] += 1
            if candidate is not None:
                break
        if candidate is None:
            failed += 1
            outcomes.append(0)
            temperature = max(np.finfo(float).tiny, temperature*rate)
            continue
        evaluations += 1
        delta = (sa_fitness(candidate.result, inst, config.alpha, penalty)
                 - sa_fitness(current.result, inst, config.alpha, penalty))
        take = delta <= 0 or (not config.hill_climbing and
                              rng.random() < math.exp(-delta/max(temperature, 1e-300)))
        recent[op].append(int(take and delta < -1e-12))
        outcomes.append(int(take))
        if take:
            current = candidate
            accepted += 1
            stats[op]['accept'] += 1
            stats[op]['improve'] += int(delta < -1e-12)
        stagnant += 1
        # Inspect every evaluated proposal, even if a low penalty rejected a feasible elite.
        if ranking(candidate.result, inst, config.alpha) < ranking(best.result, inst, config.alpha):
            best = candidate
            stagnant = 0
        infeasible_steps = infeasible_steps+1 if current.result.violation > 1e-9 else 0
        if config.adaptive_penalty and infeasible_steps >= config.stagnation_window:
            penalty = max(penalty, min(max(1., penalty*2), 1e12))
            infeasible_steps = 0
        temperature = max(np.finfo(float).tiny, temperature*rate)
        if config.reheat and stagnant >= config.stagnation_window:
            current, temperature, stagnant = best, t0, 0
    if failed:
        logging.warning('%s iterations exhausted move retries in chain %s', failed, chain)
    best_x = encode_routes(best.routes, inst)
    checked = evaluate(best_x, inst)
    if not np.allclose([checked.f1, checked.f2, checked.violation],
                       [best.result.f1, best.result.f2, best.result.violation], atol=1e-9, rtol=0):
        raise AssertionError('incremental result disagrees with independent evaluation')
    return SAResult(initial_x, initial, best_x, checked, np.array(best_hist),
                    np.array(current_hist), accepted, config.iterations,
                    np.array(temp_hist), np.array(acceptance_hist), stats,
                    evaluations, feasible=checked.violation <= 1e-9,
                    config=config.to_dict(), seed=seed, best_chain=chain,
                    failed_moves=failed, snapshots=snapshots)


def run(inst, config, seed):
    """Run independent seeded chains and report aggregate work with winner histories."""
    started = time.perf_counter()
    combined_objective(evaluate(np.full(2*inst.n, .5), inst), inst, config.alpha)
    chains = [run_chain(inst, config, np.random.default_rng(child), seed, i)
              for i, child in enumerate(np.random.SeedSequence(seed).spawn(config.n_restarts))]
    best = min(chains, key=lambda r: ranking(r.best_result, inst, config.alpha))
    best.evaluations = sum(r.evaluations for r in chains)
    best.wall_time = time.perf_counter()-started
    return best
