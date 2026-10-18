"""Exact gaps and paired baseline comparisons without external statistics packages."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from math import comb
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments import run_one, summarize, write_csv, write_json
from fitness import combined_objective, evaluate
from instance import generate_instance
from run_cases import CASES, make_case
from sa import SAConfig, _construct
from sa_engine import ranking
from validation.exact import exact_solve


def search_baseline(inst, seed, budget, greedy=False):
    """Use the same number of complete-candidate evaluations as the SA chain.

    Construction's partial insertion metric calls are additional work, so wall
    times must also be reported; equal candidate counts do not mean equal CPU.
    """
    start = perf_counter()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(budget):
        x = _construct(inst, rng) if greedy else rng.random(2*inst.n)
        r = evaluate(x, inst)
        if best is None or ranking(r, inst, .5) < ranking(best, inst, .5):
            best = r
    return dict(seed=seed, F=combined_objective(best, inst), makespan=best.f1,
                energy=best.f2, runtime=perf_counter()-start, feasible=best.violation <= 1e-9,
                violation=best.violation, evaluations=budget)


def paired(sa, other, seed=123):
    """Exact two-sided sign test on feasibility-first wins plus paired F bootstrap."""
    def compare(a, b):
        """Use a numerical tolerance so roundoff cannot become a statistical win."""
        if a['feasible'] != b['feasible']:
            return -1 if a['feasible'] else 1
        if not a['feasible'] and abs(a['violation']-b['violation']) > 1e-9:
            return -1 if a['violation'] < b['violation'] else 1
        if abs(a['F']-b['F']) <= 1e-9:
            return 0
        return -1 if a['F'] < b['F'] else 1
    signs = [compare(a, b) for a, b in zip(sa, other)]
    wins = signs.count(-1)
    losses = signs.count(1)
    n = wins+losses
    p = min(1., 2*sum(comb(n, k) for k in range(min(wins, losses)+1))/2**n) if n else 1.
    differences = np.array([a['F']-b['F'] for a, b in zip(sa, other) if a['feasible'] and b['feasible']])
    ci = None
    if differences.size:
        rng = np.random.default_rng(seed)
        means = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return dict(wins=wins, losses=losses, ties=len(sa)-n, sign_test_p=p,
                mean_F_difference=float(differences.mean()) if differences.size else None,
                paired_feasible_count=len(differences), bootstrap_95_ci=ci, bootstrap_seed=seed)


def baseline_job(job):
    """Run all algorithms for one seed in a separate standard-library worker."""
    inst, config, seed = job
    sa_row, _ = run_one((inst, config, seed))
    budget = sa_row['evaluations']
    return dict(sa=sa_row, random=search_baseline(inst, seed, budget),
                greedy=search_baseline(inst, seed, budget, greedy=True),
                hill=run_one((inst, replace(config, hill_climbing=True), seed))[0])


def main():
    """Run exact tiny tests and same-candidate-budget baseline experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('results/validation'))
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gaps = []
    for n in (6, 7, 8, 9):
        inst = generate_instance(n=n, m_g=1 if n < 8 else 2, m_a=1 if n < 8 else 2, seed=10+n, t_max=1800)
        optimum, exact_time = exact_solve(inst)
        if optimum is None:
            raise AssertionError('tiny validation instance is infeasible')
        optimum_f = combined_objective(optimum, inst)
        config = SAConfig(iterations=args.iterations, construction_restarts=10)
        rows = [run_one((inst, config, seed))[0] for seed in range(args.seeds)]
        gap = np.array([(r['F']/optimum_f-1)*100 for r in rows if r['feasible']])
        summary = dict(n=n, instance_seed=10+n, exact_F=optimum_f, exact_seconds=exact_time,
                       gap_best=float(gap.min()), gap_mean=float(gap.mean()), gap_worst=float(gap.max()),
                       hit_fraction=sum(r['feasible'] and abs(r['F']-optimum_f)<1e-9 for r in rows)/len(rows),
                       sa_seconds_mean=float(np.mean([r['runtime'] for r in rows])))
        gaps.append(summary)
        write_csv(args.output/f'exact_n{n}_runs.csv', rows)
        write_json(args.output/f'exact_n{n}.json', dict(summary=summary, routes=optimum.routes, config=config.to_dict()))
        write_csv(args.output/'exact_gaps.csv', gaps)
        print(summary, flush=True)
    if args.skip_baselines:
        return
    summaries, comparisons = [], []
    for name in CASES:
        inst = make_case(name)
        config = SAConfig(iterations=args.iterations, construction_restarts=1)
        all_rows = {name: [] for name in ('sa', 'random', 'greedy', 'hill')}
        jobs = [(inst, config, seed) for seed in range(args.seeds)]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                outcomes = list(pool.map(baseline_job, jobs))
        else:
            outcomes = list(map(baseline_job, jobs))
        for outcome in outcomes:
            for algorithm, row in outcome.items():
                all_rows[algorithm].append(row)
        print(name, 'all paired seeds complete', flush=True)
        for algorithm, rows in all_rows.items():
            write_csv(args.output/f'{name}_{algorithm}.csv', rows)
            summaries.append(dict(case=name, algorithm=algorithm, **summarize(rows)))
            if algorithm != 'sa':
                comparisons.append(dict(case=name, baseline=algorithm, **paired(all_rows['sa'], rows)))
        write_csv(args.output/'baseline_summary.csv', summaries)
        write_json(args.output/'paired_comparisons.json', comparisons)


if __name__ == '__main__':
    main()
