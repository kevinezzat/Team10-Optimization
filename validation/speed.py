"""Time identical route-update replays through cached and legacy full evaluation."""
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments import write_csv
from fitness import decode, evaluate
from instance import generate_instance
from sa import SAConfig, encode_routes, initial_solution
from sa_engine import RouteState, propose


def benchmark(n, repeats=3, moves=3000):
    """Exclude move generation from both timings to isolate evaluation overhead."""
    inst = generate_instance(n=n, m_g=5, m_a=5, seed=0, t_max=1800)
    rng = np.random.default_rng(5)
    x, _ = initial_solution(inst, rng, 1, .5, 100)
    initial = decode(x, inst)
    state = RouteState(initial, inst)
    updates = []
    for _ in range(moves):
        new = propose(state, inst, rng, str(rng.choice(SAConfig().operators)))
        if new is not None:
            updates.append({k: route for k, route in enumerate(new.routes) if route != state.routes[k]})
            state = new
    full_times, cache_times = [], []
    for _ in range(repeats):
        state = RouteState(initial, inst)
        start = perf_counter()
        for change in updates:
            state = state.changed(change, inst)
        cache_times.append(perf_counter()-start)
        current_x = x.copy()
        start = perf_counter()
        for change in updates:
            routes = [r.copy() for r in decode(current_x, inst)]
            for k, route in change.items():
                routes[k] = route
            current_x = encode_routes(routes, inst)
            full = evaluate(current_x, inst)
        full_times.append(perf_counter()-start)
        np.testing.assert_allclose(full[:3], state.result[:3], atol=1e-9, rtol=0)
    return dict(n=n, seed=5, moves=len(updates), repeats=repeats,
                full_seconds=float(np.median(full_times)), cached_seconds=float(np.median(cache_times)),
                speedup=float(np.median(full_times)/np.median(cache_times)))


def main():
    """Persist paired replay timing for n=50 and n=100."""
    rows = [benchmark(n) for n in (50, 100)]
    Path('results/validation').mkdir(parents=True, exist_ok=True)
    write_csv('results/validation/speed.csv', rows)
    print(rows)


if __name__ == '__main__':
    main()
