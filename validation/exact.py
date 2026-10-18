"""Exact subset tour enumeration and vehicle partition search (n <= 9)."""
from itertools import permutations
from time import perf_counter
from fitness import combined_objective, evaluate, route_metrics
from sa import compatible, encode_routes


def exact_solve(inst, alpha=.5):
    """Return the globally optimal feasible result and elapsed seconds.

    For each vehicle/subset, a shortest tour dominates every longer ordering:
    service and payload are fixed, while time and energy increase with distance.
    Exhaustively partitioning these subsets therefore proves global optimality.
    """
    if not 1 <= inst.n <= 9:
        raise ValueError('exact solver supports 1..9 tasks')
    start = perf_counter()
    tours = []
    for k in range(inst.m):
        table = {0: ([], dict(dist=0., time=0., energy=0., load=0.))}
        for mask in range(1, 1 << inst.n):
            tasks = [i+1 for i in range(inst.n) if mask >> i & 1]
            if not all(compatible(inst, k, t) for t in tasks):
                continue
            if sum(inst.demand[t-1] for t in tasks) > inst.payload[k]+1e-12:
                continue
            d = inst.dist(k)
            route = min(permutations(tasks), key=lambda r:
                        d[0, r[0]] + sum(d[a, b] for a, b in zip(r, r[1:])) + d[r[-1], 0])
            metrics = route_metrics(k, route, inst)
            if metrics['energy'] <= inst.battery[k]+1e-12 and metrics['time'] <= inst.t_max+1e-12:
                table[mask] = (list(route), metrics)
        tours.append(table)
    best, best_cost = None, float('inf')

    def partition(k, remaining, routes):
        """Enumerate disjoint feasible subsets for all labeled vehicles."""
        nonlocal best, best_cost
        if k == inst.m:
            if remaining:
                return
            result = evaluate(encode_routes(routes, inst), inst)
            cost = combined_objective(result, inst, alpha)
            if cost < best_cost:
                best, best_cost = result, cost
            return
        subset = remaining
        while True:
            if subset in tours[k]:
                partition(k+1, remaining ^ subset, routes+[tours[k][subset][0]])
            if subset == 0:
                break
            subset = (subset-1) & remaining

    partition(0, (1 << inst.n)-1, [])
    return best, perf_counter()-start
