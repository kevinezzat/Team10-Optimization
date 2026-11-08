"""Build a readable local HTML report from measured results and methodology."""
import html
import json
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'results/milestone4'
data=json.loads((out/'summary.json').read_text())
def fmt(v): return 'No feasible result' if v is None else f'{v:.4f}'
parts=['<!doctype html><meta charset="utf-8"><title>Team 10 — Milestone 4</title>',
'<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:50px auto;padding:20px;color:#172439}h1,h2{color:#183e66}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ccd6df;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f6f9;padding:20px}</style>',
'<h1>Team 10 — Genetic Algorithm and SA comparison</h1><p>Milestone 4 working report · 8 November 2026</p>',
'<h2>Problem and implementation</h2><p>Allocate and sequence tasks for a mixed UAV/UGV fleet. Minimize mission makespan and total energy while enforcing capability, payload, battery and deadline constraints. Both solvers use the same normalized scalar objective with equal time and energy weights.</p>',
'<p>The GA uses 60 random-key chromosomes, three-way tournaments, uniform crossover (0.9), per-gene reset mutation (0.05), capability repair and two elites. Half the initial population uses randomized insertion. Feasible candidates outrank infeasible candidates. Every evaluated chromosome counts toward the budget.</p>',
'<h2>Experiment design</h2><p>Five solver seeds per configuration, fixed problem seed zero and 3000 evaluation upper bound. SA construction and calibration are included. SA can use fewer evaluations when proposals fail, and caches route calculations. Runtime is therefore reported alongside work counts. Exact validation cases have a 10000-second deadline; scale cases retain their prior deadlines. Full inputs and per-run routes are supplied in JSON.</p>',
'<h2>Results</h2><img src="comparison.png" alt="GA and SA comparison"><table><tr><th>Case</th><th>Solver</th><th>Feasible</th><th>Mean F</th><th>Makespan (s)</th><th>Energy (kJ)</th><th>Runtime (s)</th><th>Gap (%)</th></tr>']
for s in data:
    parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [s['case'],s['algorithm'],f"{s['feasibility_rate']:.0%}",*(fmt(s[k+'_mean']) for k in ('objective','makespan','energy','seconds','gap_percent'))])+'</tr>')
parts += ['</table><p>Means for objective, mission time and energy include feasible runs only. The raw CSV includes all infeasible outcomes and their constraint violations. Sample standard deviations are provided in summary.json.</p>',
'<h2>Convergence, parameter effects and routes</h2><img src="ga_convergence.png" alt="GA convergence"><img src="parameters.png" alt="Parameter effects"><img src="reference30_GA_routes.png" alt="GA routes"><img src="reference30_SA_routes.png" alt="SA routes">',
'<h2>Interpretation and limitations</h2><p>Assess feasibility before comparing objectives. Compare makespan and energy as the two primary problem-specific indices, with runtime and robustness as additional indices. Exact gaps certify small-case solution quality; no optimality is claimed for larger cases. Repeated solver seeds on one problem instance measure search variation, not generalization across independent instances. Synthetic scale cases are not published benchmark reproductions.</p>',
'<p>Population, crossover and mutation are varied one factor at a time. A larger population has fewer generations under a fixed evaluation budget. These studies do not identify interactions or a universal parameter choice. The implementation uses NumPy and Matplotlib, which the team confirmed are allowed.</p>',
'<h2>Reproduction and recording notes</h2><pre>'+html.escape((root/'MILESTONE4_GUIDE.md').read_text())+'</pre>']
(out/'REPORT.html').write_text('\n'.join(parts),encoding='utf-8')

