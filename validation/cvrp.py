"""Verifiable CVRP reduction using embedded synthetic data and exact optimum."""
import json
from pathlib import Path
import numpy as np
from instance import generate_instance
from sa import SAConfig, simulated_annealing
from validation.exact import exact_solve
from experiments import write_json


def load_cvrp(path=None):
    """Build identical vehicles; provable upper bounds make time/energy nonbinding.

    With alpha=0 and e=1, report F is a positive constant times total distance.
    Finite nonbinding bounds preserve the report evaluator's public contract.
    """
    path = path or Path(__file__).parent/'data/cvrp_tiny.json'
    data = json.loads(Path(path).read_text())
    coords = np.asarray(data['coordinates'], dtype=float)
    n = len(coords)-1
    inst = generate_instance(n=n, m_g=data['vehicles'], m_a=0,
                             p_ugv_only=0, p_uav_only=0, seed=0, t_max=1)
    inst.coords = coords
    inst.D_ugv = inst.D_uav = np.linalg.norm(coords[:, None]-coords[None, :], axis=2)
    inst.demand = np.asarray(data['demands'], dtype=float)
    inst.service[:] = 0
    inst.speed[:] = inst.e_rate[:] = 1
    inst.payload[:] = data['capacity']
    # Every route has at most n+1 edges, each no longer than max(D).
    bound = (n+1)*inst.D_ugv.max()+1
    inst.battery[:] = bound
    inst.t_max = bound
    return inst, data


def main():
    """Compare 20 SA runs with the exact synthetic CVRP optimum."""
    inst, data = load_cvrp()
    optimum, elapsed = exact_solve(inst, alpha=0)
    rows = []
    for seed in range(20):
        result = simulated_annealing(inst, SAConfig(iterations=2000, construction_restarts=10, alpha=0), seed)
        rows.append(dict(seed=seed, distance=result.best_result.f2,
                         gap_percent=100*(result.best_result.f2/optimum.f2-1), feasible=result.feasible))
    directory = Path('results/validation')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'cvrp.json', dict(source=data, exact_distance=optimum.f2,
               exact_seconds=elapsed, exact_routes=optimum.routes, runs=rows))
    print('Synthetic CVRP exact distance:', optimum.f2, 'mean gap:', np.mean([r['gap_percent'] for r in rows]))


if __name__ == '__main__':
    main()
