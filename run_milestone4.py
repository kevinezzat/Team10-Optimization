"""Reproduce GA studies and SA comparison: python run_milestone4.py."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ga import GAConfig, genetic_algorithm
from sa import SAConfig, simulated_annealing
from fitness import combined_objective
from instance import generate_instance
from run_cases import make_case
from validation.exact import exact_solve
from visualize import plot_routes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--budget', type=int, default=3000)
    p.add_argument('--output', type=Path, default=Path('results/milestone4'))
    args = p.parse_args()
    if args.seeds < 1 or args.budget < 120:
        p.error('positive seeds and budget >=120 required')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cfg = GAConfig(evaluations=args.budget)
    def run(name, inst, algorithm, seed, config=cfg, optimum=None):
        instance_file=out/(name+'_instance.json')
        if not instance_file.exists():
            instance_file.write_text(json.dumps({k: v.tolist() if isinstance(v,np.ndarray) else v
                for k,v in vars(inst).items()},indent=2))
        if algorithm == 'SA':
            # Initialization + calibration + proposals <= requested budget.
            r = simulated_annealing(inst, SAConfig(iterations=args.budget-80,
                construction_restarts=60, calibration_samples=20), seed)
            result, elapsed, evaluations = r.best_result, r.wall_time, r.evaluations
        else:
            r = genetic_algorithm(inst, config, seed)
            result, elapsed, evaluations = r['best_result'], r['wall_time'], r['evaluations']
            (out/(name+'_GA_'+str(seed)+'_history.json')).write_text(json.dumps(r['history']))
        feasible = result.violation <= 1e-9
        objective = combined_objective(result, inst)
        row = dict(case=name, algorithm=algorithm, seed=seed, feasible=feasible,
            objective=objective if feasible else None, makespan=result.f1,
            energy=result.f2, violation=result.violation, seconds=elapsed,
            evaluations=evaluations, gap_percent=100*(objective/optimum-1)
            if feasible and optimum is not None else None)
        rows.append(row)
        (out/(name+'_'+algorithm+'_'+str(seed)+'.json')).write_text(json.dumps(
            dict(metrics=row, routes=result.routes, config=r.config if algorithm=='SA' else r['config']), indent=2))
        if seed == 0:
            plot_routes(inst, result, out/(name+'_'+algorithm+'_routes.png'))
        print(name, algorithm, seed, 'feasible=', feasible, 'F=', round(objective,5), flush=True)
        return r
    cases = {name: make_case(name) for name in ('small10','reference30','gao50','battery75','restricted90','gao100')}
    for n in (6,7):
        inst = generate_instance(n=n, m_g=1, m_a=1, seed=0, t_max=10000.)
        exact, elapsed = exact_solve(inst)
        if exact is None:
            raise RuntimeError('exact validation instance infeasible')
        optimum = combined_objective(exact, inst)
        (out/f'exact_n{n}.json').write_text(json.dumps(dict(objective=optimum, routes=exact.routes, seconds=elapsed), indent=2))
        for seed in range(args.seeds):
            for algo in ('GA','SA'):
                run(f'exact{n}', inst, algo, seed, optimum=optimum)
    for name, inst in cases.items():
        for seed in range(args.seeds):
            for algo in ('GA','SA'):
                run(name, inst, algo, seed)
    for field, values in [('population',(30,60,120)), ('crossover',(.6,.9,1.)), ('mutation',(.01,.05,.15))]:
        for value in values:
            for seed in range(args.seeds):
                run(f'{field}_{value}', cases['reference30'], 'GA', seed, replace(cfg, **{field:value}))
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summaries=[]
    for name, algo in sorted({(r['case'],r['algorithm']) for r in rows}):
        group=[r for r in rows if r['case']==name and r['algorithm']==algo]
        feasible=[r for r in group if r['feasible']]
        s=dict(case=name, algorithm=algo, runs=len(group), feasibility_rate=len(feasible)/len(group))
        for field in ('objective','makespan','energy','seconds','evaluations','gap_percent'):
            vals=[r[field] for r in (group if field in ('seconds','evaluations') else feasible) if r[field] is not None]
            s[field+'_mean']=float(np.mean(vals)) if vals else None
            s[field+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summaries.append(s)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    names=list(cases)
    for ax,metric,title in zip(axes,('objective','seconds','feasibility_rate'),('Feasible objective (lower is better)','Runtime [s]','Feasibility rate')):
        for offset,algo in ((-.18,'GA'),(.18,'SA')):
            data=[next(s for s in summaries if s['case']==name and s['algorithm']==algo) for name in names]
            key=metric if metric=='feasibility_rate' else metric+'_mean'
            ax.bar(np.arange(len(names))+offset,[s[key] if s[key] is not None else np.nan for s in data],width=.36,label=algo)
        ax.set_xticks(range(len(names)),names,rotation=50,ha='right'); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    text=['# Milestone 4 measured results','',f'{args.seeds} seeds per configuration; evaluation budget {args.budget}. Instance seed 0. Alpha 0.5.','',
          '| Case | Algorithm | Feasible | Mean F | Mean time (s) |','|---|---|---:|---:|---:|']
    for s in summaries:
        text.append(f"| {s['case']} | {s['algorithm']} | {s['feasibility_rate']:.0%} | {s['objective_mean']} | {s['seconds_mean']:.3f} |")
    text += ['', 'Objectives, makespan and energy are summarized over feasible runs only. Infeasible runs remain in runs.csv with their violation.',
      'GA uses its full evaluation budget. SA has the same upper bound but may use fewer evaluations when proposal generation fails. Runtime includes construction. Evaluation counts are not identical costs: SA caches unchanged routes.',
      'These are synthetic instances, not reproduced literature datasets. Exact small-instance optima provide validation. No universal superiority claim is supported. Parameter studies hold the evaluation budget fixed; larger populations therefore receive fewer generations.']
    (out/'RESULTS.md').write_text('\n'.join(text))


if __name__ == '__main__':
    main()
