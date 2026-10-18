"""Run Milestone 3 Simulated Annealing on the existing reproducible instance."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitness import combined_objective
from instance import ANY, UAV_ONLY, UGV_ONLY, generate_instance
from sa import SAConfig, simulated_annealing


from visualize import plot_routes, plot_convergence, diagnostics, snapshots


def main():
    """Run and save one reproducible annealing experiment."""
    parser = argparse.ArgumentParser(description='Team10 Milestone 3: Simulated Annealing')
    parser.add_argument('--tasks', type=int, default=30)
    parser.add_argument('--ugvs', type=int, default=2)
    parser.add_argument('--uavs', type=int, default=3)
    parser.add_argument('--instance-seed', type=int, default=0)
    parser.add_argument('--sa-seed', type=int, default=1)
    parser.add_argument('--deadline', type=float, default=1800.0, help='T_max in seconds')
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--restarts', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--final-temperature', type=float, default=1e-5)
    parser.add_argument('--target-acceptance', type=float, default=.8)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--penalty', type=float, default=100.)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument('--adaptive-operators', action='store_true')
    parser.add_argument('--reheat', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs_sa_n30'))
    args = parser.parse_args()
    if args.tasks <= 0 or args.ugvs <= 0 or args.uavs <= 0 or args.deadline <= 0:
        parser.error('tasks, ugvs, uavs, deadline must all be positive')

    inst = generate_instance(n=args.tasks, m_g=args.ugvs, m_a=args.uavs,
                             seed=args.instance_seed, t_max=args.deadline)
    config = SAConfig(iterations=args.iterations, construction_restarts=args.restarts, initial_temperature=args.temperature, final_temperature=args.final_temperature, target_acceptance=args.target_acceptance, alpha=args.alpha, penalty=args.penalty, n_restarts=args.chains, adaptive_operators=args.adaptive_operators, reheat=args.reheat)
    tic = time.perf_counter()
    res = simulated_annealing(inst, config, seed=args.sa_seed)
    elapsed = time.perf_counter() - tic
    initial_F = combined_objective(res.initial_result, inst, config.alpha)
    best_F = combined_objective(res.best_result, inst, config.alpha)
    improvement = 100 * (initial_F - best_F) / initial_F if initial_F > 0 else 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics(res, args.output)
    snapshots(inst, res, args.output / 'frames')
    plot_convergence(res, args.output / 'convergence.png')
    plot_routes(inst, res.best_result, args.output / 'routes.png')
    metrics = {
        'instance': {'n': inst.n, 'ugvs': inst.m_g, 'uavs': inst.m_a,
                     'instance_seed': args.instance_seed, 'sa_seed': args.sa_seed,
                     'deadline_seconds': inst.t_max,
                     'battery_kJ_by_vehicle': inst.battery.tolist()},
        'sa_config': config.to_dict(),
        'results': {'feasible': res.feasible, 'evaluations': res.evaluations, 'best_chain': res.best_chain, 'operator_stats': res.operator_stats, 'initial_F': initial_F, 'best_F': best_F,
                    'improvement_percent': improvement,
                    'makespan_seconds': res.best_result.f1,
                    'energy_kJ': res.best_result.f2,
                    'violation': res.best_result.violation,
                    'accepted_moves': res.accepted,
                    'attempted_moves': res.attempted,
                    'execution_seconds': elapsed,
                    'routes': res.best_result.routes,
                    'constraints': {k: float(v) for k, v in
                                    res.best_result.details.items() if k.startswith('g_')}}}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f"Scenario: {inst.n} tasks / {inst.m_g} UGVs / {inst.m_a} UAVs")
    print(f"Initial F: {initial_F:.6f} (violation={res.initial_result.violation:.6f})")
    print(f"Best returned F: {best_F:.6f} | improvement: {improvement:.2f}%")
    print(f"Makespan: {res.best_result.f1:.2f} s | energy: {res.best_result.f2:.2f} kJ")
    print(f"Violations: {res.best_result.violation:.6f} | accepted: {res.accepted}/{res.attempted}")
    print(f"Runtime: {elapsed:.2f} s | saved results to {args.output}")


if __name__ == '__main__':
    main()
