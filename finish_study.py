"""Add parameter plots and measured interpretation to the study report."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('results/milestone4')
data=json.loads((out/'summary.json').read_text())
paragraphs=['## Interpretation of measured results','']
for name in ('small10','reference30','gao50','battery75','restricted90','gao100'):
    ga=next(s for s in data if s['case']==name and s['algorithm']=='GA')
    sa=next(s for s in data if s['case']==name and s['algorithm']=='SA')
    if ga['objective_mean'] is not None and sa['objective_mean'] is not None:
        difference=100*(ga['objective_mean']/sa['objective_mean']-1)
        paragraphs.append(f"{name}: mean feasible GA objective {ga['objective_mean']:.5f}, SA {sa['objective_mean']:.5f}; GA is {abs(difference):.2f}% {'higher' if difference>0 else 'lower'}. Feasibility rates: GA {ga['feasibility_rate']:.0%}, SA {sa['feasibility_rate']:.0%}.")
    else:
        paragraphs.append(f"{name}: GA feasibility {ga['feasibility_rate']:.0%}, SA {sa['feasibility_rate']:.0%}. No objective comparison is made without feasible solutions; failure to find a feasible solution is not a proof of infeasibility.")
    paragraphs.append('')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,field in zip(axes,('population','crossover','mutation')):
    group=sorted([s for s in data if s['case'].startswith(field+'_')],key=lambda s:float(s['case'].split('_')[1]))
    ax.errorbar([float(s['case'].split('_')[1]) for s in group],
        [s['objective_mean'] for s in group],yerr=[s['objective_std'] or 0 for s in group],marker='o',capsize=4)
    ax.set(xlabel=field,ylabel='Mean feasible F +/- sample SD'); ax.grid(alpha=.2)
    best=min(group,key=lambda s:float('inf') if s['objective_mean'] is None else s['objective_mean'])
    paragraphs += [f"Among tested {field} values, {best['case'].split('_')[1]} had the lowest mean objective ({best['objective_mean']:.5f}) on the 30-task instance. This is descriptive evidence from five seeds, not an independently validated tuning recommendation.",'']
fig.tight_layout(); fig.savefig(out/'parameters.png',dpi=160)
with (out/'RESULTS.md').open('a') as f: f.write('\n\n'+'\n'.join(paragraphs))
