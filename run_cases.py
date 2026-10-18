"""Named scale/constraint case studies with process-based seeded repeats."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from instance import generate_instance
from sa import SAConfig
from experiments import instance_metadata, run_one, summarize, write_csv, write_json
from visualize import diagnostics, plot_convergence, plot_routes, save
import matplotlib.pyplot as plt

CASES = {
    'small10': (10, 1, 1, 1800., .2, .2, 1.),
    'chen15': (15, 2, 2, 1800., .2, .2, 1.),
    'reference30': (30, 2, 3, 1800., .2, .2, 1.),
    'gao50': (50, 3, 3, 1200., .2, .2, 1.),
    'battery75': (75, 4, 4, 1800., .2, 0., .12),
    'restricted90': (90, 5, 5, 1800., .45, .45, 1.),
    'gao100': (100, 5, 5, 1800., .2, .2, 1.),
}


def make_case(name, seed=0):
    """Generate a named case; battery stress has no mandatory aerial tasks."""
    n, g, a, deadline, pg, pa, battery_factor = CASES[name]
    inst = generate_instance(n=n, m_g=g, m_a=a, t_max=deadline, seed=seed,
                             p_ugv_only=pg, p_uav_only=pa)
    inst.battery[g:] *= battery_factor
    return inst


def main():
    """Run all configured cases, persist raw replicates, and summarize scaling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--restarts', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--output', type=Path, default=Path('results/cases'))
    args = parser.parse_args()
    if args.seeds < 1 or args.workers < 1:
        parser.error('seeds and workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for name in args.cases:
        inst = make_case(name)
        directory = args.output/name
        directory.mkdir(exist_ok=True)
        config = SAConfig(iterations=args.iterations, construction_restarts=args.restarts)
        jobs = [(inst, config, seed) for seed in range(args.seeds)]
        if args.workers == 1:
            outcomes = list(map(run_one, jobs))
        else:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    outcomes = list(pool.map(run_one, jobs))
            except (OSError, PermissionError) as error:
                print(f'Process startup unavailable ({error}); using sequential execution.', flush=True)
                outcomes = list(map(run_one, jobs))
        rows, results = map(list, zip(*outcomes))
        summary = dict(case=name, n=inst.n, ugvs=inst.m_g, uavs=inst.m_a, **summarize(rows))
        aggregate.append(summary)
        best = min(results, key=lambda r: (not r.feasible, r.best_result.violation, r.best_history[-1] if r.feasible else 0))
        write_csv(directory/'runs.csv', rows)
        write_csv(directory/'summary.csv', [summary])
        params = dict(n=inst.n, ugvs=inst.m_g, uavs=inst.m_a, deadline=inst.t_max,
                      battery=inst.battery.tolist(), payload=inst.payload.tolist(),
                      speed=inst.speed.tolist(), energy_rate=inst.e_rate.tolist(),
                      task_mix=np.bincount(inst.ttype, minlength=3).tolist(), instance_seed=0)
        write_csv(directory/'parameters.csv', [params])
        write_json(directory/'metrics.json', dict(instance=instance_metadata(inst, 0),
                   config=config.to_dict(), runs=rows, best_routes=best.best_result.routes,
                   best_seed=best.seed, summary=summary))
        plot_routes(inst, best.best_result, directory/'routes.png')
        plot_convergence(results, directory/'convergence.png')
        diagnostics(best, directory)
        print(name, params, summary, flush=True)
        write_csv(args.output/'all_cases_summary.csv', aggregate)
    lines = ['| Case | n | Fleet G+A | F mean ± SD | Feasible | Runtime mean (s) |',
             '|---|---:|---:|---:|---:|---:|']
    for row in aggregate:
        quality = 'no feasible runs' if row['F_mean'] is None else f"{row['F_mean']:.6f} ± {row['F_std']:.6f}"
        lines.append(f"| {row['case']} | {row['n']} | {row['ugvs']}+{row['uavs']} | {quality} | {row['feasibility_rate']:.0%} | {row['runtime_mean']:.2f} |")
    (args.output/'report_table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    n = np.array([r['n'] for r in aggregate])
    runtime = np.array([r['runtime_mean'] for r in aggregate])
    axes[0].plot(n, runtime, 'o-')
    axes[0].set(xlabel='Tasks', ylabel='Runtime [s]')
    axes[1].plot(n, [r['F_mean'] for r in aggregate], 'o-')
    axes[1].set(xlabel='Tasks', ylabel='Mean feasible F')
    save(fig, args.output/'scalability.png')
    if len(n)>1:
        slope, intercept = np.polyfit(np.log(n), np.log(runtime), 1)
        write_json(args.output/'runtime_fit.json', dict(exponent=slope, intercept=intercept,
                   caveat='Descriptive fit: fleet and constraints vary with n; not isolated complexity.'))


if __name__ == '__main__':
    main()
