"""Produce a reproducible example convergence trace (separate from study rows)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ga import genetic_algorithm, GAConfig
from run_cases import make_case

out=Path('results/milestone4')
r=genetic_algorithm(make_case('reference30'),GAConfig(evaluations=3000),seed=0)
(out/'example_convergence.json').write_text(json.dumps(r['history']))
fig,ax=plt.subplots(figsize=(7,4))
ax.plot([x for x,y in r['history']],[float('nan') if y is None else y for x,y in r['history']])
ax.set(xlabel='Fitness evaluations (including initialization)',ylabel='Best feasible objective',title='GA convergence: 30 tasks, seed 0')
ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'ga_convergence.png',dpi=160)
