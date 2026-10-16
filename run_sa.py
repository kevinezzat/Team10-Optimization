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


def plot_routes(inst, result, destination):
    fig, ax = plt.subplots(figsize=(8, 7))
    markers = {ANY: 'o', UGV_ONLY: 's', UAV_ONLY: '^'}
    labels = {ANY: 'Any', UGV_ONLY: 'UGV only', UAV_ONLY: 'UAV only'}
    for kind in (ANY, UGV_ONLY, UAV_ONLY):
        ids = np.where(inst.ttype == kind)[0] + 1
        if len(ids):
            ax.scatter(inst.coords[ids, 0], inst.coords[ids, 1], marker=markers[kind],
                       c='gray', edgecolors='black', s=55, label=labels[kind])
            for i in ids:
                ax.annotate(str(i), inst.coords[i], xytext=(4, 4),
                            textcoords='offset points', fontsize=7)
    ax.scatter(*inst.coords[0], marker='*', s=270, color='gold',
               edgecolors='black', label='Depot', zorder=4)
    for k, route in enumerate(result.routes):
        if not route:
            continue
        pts = inst.coords[np.r_[0, route, 0]]
        is_uav = inst.is_uav(k)
        ax.plot(pts[:, 0], pts[:, 1], '--' if is_uav else '-',
                linewidth=1.8, color=plt.cm.tab10(k % 10),
                label=f"{'UAV' if is_uav else 'UGV'} {k - inst.m_g + 1 if is_uav else k + 1}")
    ax.set(xlabel='x [m]', ylabel='y [m]',
           title=f'SA routes | makespan {result.f1:.1f} s | energy {result.f2:.1f} kJ')
    ax.axis('equal'); ax.legend(fontsize=8, loc='best')
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def plot_convergence(res, destination):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(res.best_history, label='Best feasible objective F', linewidth=1.7)
    ax.set(xlabel='Iteration', ylabel='Normalized weighted objective F',
           title='Simulated Annealing convergence')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Team10 Milestone 3: Simulated Annealing')
    parser.add_argument('--tasks', type=int, default=30)
    parser.add_argument('--ugvs', type=int, default=2)
    parser.add_argument('--uavs', type=int, default=3)
    parser.add_argument('--instance-seed', type=int, default=0)
    parser.add_argument('--sa-seed', type=int, default=1)
    parser.add_argument('--deadline', type=float, default=1800.0, help='T_max in seconds')
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--restarts', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('outputs_sa_n30'))
    args = parser.parse_args()
    if args.tasks <= 0 or args.ugvs <= 0 or args.uavs <= 0 or args.deadline <= 0:
        parser.error('tasks, ugvs, uavs, deadline must all be positive')

    inst = generate_instance(n=args.tasks, m_g=args.ugvs, m_a=args.uavs,
                             seed=args.instance_seed, t_max=args.deadline)
    config = SAConfig(iterations=args.iterations, construction_restarts=args.restarts)
    tic = time.perf_counter()
    res = simulated_annealing(inst, config, seed=args.sa_seed)
    elapsed = time.perf_counter() - tic
    initial_F = combined_objective(res.initial_result, inst)
    best_F = combined_objective(res.best_result, inst)
    improvement = 100 * (initial_F - best_F) / initial_F if initial_F > 0 else 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    plot_convergence(res, args.output / 'convergence.png')
    plot_routes(inst, res.best_result, args.output / 'routes.png')
    metrics = {
        'instance': {'n': inst.n, 'ugvs': inst.m_g, 'uavs': inst.m_a,
                     'instance_seed': args.instance_seed, 'sa_seed': args.sa_seed,
                     'deadline_seconds': inst.t_max,
                     'battery_kJ_by_vehicle': inst.battery.tolist()},
        'sa_config': {'iterations': config.iterations,
                      'initial_temperature': config.initial_temperature,
                      'cooling_rate': config.cooling_rate,
                      'penalty': config.penalty, 'alpha': config.alpha,
                      'construction_restarts': config.construction_restarts},
        'results': {'initial_F': initial_F, 'best_F': best_F,
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
    print(f"Best feasible F: {best_F:.6f} | improvement: {improvement:.2f}%")
    print(f"Makespan: {res.best_result.f1:.2f} s | energy: {res.best_result.f2:.2f} kJ")
    print(f"Violations: {res.best_result.violation:.6f} | accepted: {res.accepted}/{res.attempted}")
    print(f"Runtime: {elapsed:.2f} s | saved results to {args.output}")


if __name__ == '__main__':
    main()
