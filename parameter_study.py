"""One-factor sweeps and a small full-factorial SA study."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
import numpy as np
from experiments import run_one, summarize, write_csv, write_json
from run_cases import make_case
from sa import SAConfig
from visualize import save
import matplotlib.pyplot as plt


def settings(base):
    """Yield every requested controlled sweep, plus acceptance x penalty grid."""
    factors = dict(iterations=[2000, 5000, 12000, 30000, 60000],
                   target_acceptance=[.2, .5, .8, .95],
                   final_temperature=[1e-3, 1e-5, 1e-7], penalty=[1., 10., 100., 1000.],
                   alpha=[i/10 for i in range(11)], construction_restarts=[1, 10, 100],
                   adaptive_operators=[False, True], reheat=[False, True])
    for name, values in factors.items():
        for value in values:
            yield name, str(value), replace(base, **{name: value})
    for op in base.operators:
        yield 'ablation', 'without_'+op, replace(base, operators=tuple(x for x in base.operators if x != op))
        yield 'ablation', 'only_'+op, replace(base, operators=(op,))
    for acceptance, penalty in product([.5, .8], [10., 100.]):
        yield 'factorial', f'p={acceptance},lambda={penalty}', replace(base, target_acceptance=acceptance, penalty=penalty)


def main():
    """Run paired seeds for each setting and save measured, qualified findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/parameters'))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('seeds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    inst = make_case('reference30')
    base = SAConfig(iterations=args.iterations, construction_restarts=10)
    groups = {}
    findings = ['# Measured parameter findings', '',
                'One instance (30 tasks, instance seed 0); paired seeds. SD describes run variation, not uncertainty of the mean.',
                'Alpha changes the objective itself: compare makespan and energy, not F across alpha.', '']
    for factor, value, config in settings(base):
        slug = value.replace(',', '_').replace('=', '-')
        checkpoint = args.output/f'{factor}_{slug}_checkpoint.json'
        # Completed settings are reusable; interrupted settings are rerun as a unit.
        if args.resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            if cached['config'] == json.loads(json.dumps(config.to_dict())) and len(cached['rows']) == args.seeds:
                rows = cached['rows']
            else:
                rows = None
        else:
            rows = None
        if rows is None:
            jobs = [(inst, config, seed) for seed in range(args.seeds)]
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    rows = [item[0] for item in pool.map(run_one, jobs)]
            else:
                rows = [run_one(job)[0] for job in jobs]
            write_json(checkpoint, dict(config=config.to_dict(), rows=rows))
        summary = dict(factor=factor, setting=value, **summarize(rows))
        groups.setdefault(factor, []).append(summary)
        write_csv(args.output/f'{factor}_{slug}_runs.csv', rows)
        write_json(args.output/f'{factor}_{slug}_config.json', config.to_dict())
        write_csv(args.output/f'{factor}.csv', groups[factor])
        print(factor, value, summary['F_mean'], summary['feasibility_rate'], flush=True)
    for factor, rows in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        x = np.arange(len(rows))
        means = [r['F_mean'] if r['F_mean'] is not None else np.nan for r in rows]
        axes[0].errorbar(x, means, yerr=[r['F_std'] or 0 for r in rows], fmt='o', capsize=4)
        axes[0].set(ylabel='Feasible F mean +/- SD', title=factor)
        axes[1].plot(x, [r['feasibility_rate'] for r in rows], 'o-')
        axes[1].set(ylabel='Feasibility rate', ylim=(-.05, 1.05))
        for ax in axes:
            ax.set_xticks(x, [r['setting'] for r in rows], rotation=60, ha='right')
        save(fig, args.output/f'{factor}.png')
        feasible = [r for r in rows if r['F_mean'] is not None]
        if feasible and factor != 'alpha':
            best = min(feasible, key=lambda r: (-r['feasibility_rate'], r['F_mean']))
            worst = max(feasible, key=lambda r: r['F_mean'])
            findings.append(f"- {factor}: lowest observed mean F at {best['setting']} ({best['F_mean']:.6f} ± {best['F_std']:.6f}); highest at {worst['setting']} ({worst['F_mean']:.6f}). These descriptive rankings are not significance tests.")
    alpha_rows = groups['alpha']
    points = np.array([[r['makespan_mean'], r['energy_mean']] for r in alpha_rows], dtype=float)
    nondominated = [i for i, p in enumerate(points) if np.isfinite(p).all() and not any(
        np.all(q <= p) and np.any(q < p) for q in points)]
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.scatter(points[:, 0], points[:, 1], label='Mean by alpha')
    ax.scatter(points[nondominated, 0], points[nondominated, 1], facecolors='none', edgecolors='red', s=130, label='Nondominated means')
    for row, point in zip(alpha_rows, points):
        ax.annotate(row['setting'], point, xytext=(4, 4), textcoords='offset points')
    ax.set(xlabel='Makespan [s]', ylabel='Energy [kJ]', title='Observed trade-off (means, not a proven Pareto front)')
    ax.legend()
    save(fig, args.output/'pareto.png')
    findings += ['', 'Recommendations should balance the measured runtime and feasibility against F. Keep alpha tied to mission priorities; this single-instance study does not establish universal defaults.']
    (args.output/'findings.md').write_text('\n'.join(findings)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
