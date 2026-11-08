# Milestone 4

Working extension of Team 10 repository snapshot 0bcbe28c39bcb32d48d777a59d795e33bf8b5034.
The original synced project references were not changed. This folder is a separate working copy.

## Run

Python 3.10+ with NumPy and Matplotlib (as used in Milestone 3):

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python run_milestone4.py --seeds 5 --budget 3000
```

Results are written under results/milestone4. Increase seeds and budget for a stronger final study.
No GA/SA optimization package is used. NumPy and Matplotlib are allowed, as confirmed by the team.

## Algorithm

The chromosome contains n allocation keys and n ordering keys. Decoding serves each task exactly once, with implicit depot departure and return. Half of the population uses the previous randomized insertion constructor; half uses random keys. Capability repair resamples incompatible allocation genes only. Resource constraints are evaluated, not silently relaxed.

1. Evaluate the initial population and count every evaluation.
2. Retain the best two chromosomes unchanged.
3. Select each parent by a three-way tournament.
4. Apply uniform crossover with probability 0.9.
5. Independently reset each gene with probability 0.05.
6. Repair capability, evaluate and fill the next generation.
7. Stop at the evaluation budget, including a partial final generation.

Feasible solutions rank before infeasible solutions. Feasible individuals minimize F = 0.5*makespan/deadline + 0.5*energy/total_battery. Infeasible individuals first minimize normalized violation, then F. This is a scalar GA, not a Pareto-front algorithm.

## Validation and comparison

Exact enumeration on 6- and 7-task instances provides true feasible optima. Report relative gap as 100*(F-Fopt)/Fopt. Six larger cases reuse previous milestone data generation, including battery stress and restricted capability mixes. Five repeated seeds characterize variation on each fixed instance; they do not represent five independent problem instances.

Report makespan [s], energy [kJ], feasibility rate, mean objective, sample standard deviation, runtime and evaluations. Report infeasible outcomes explicitly. SA and GA share an evaluation upper bound, with initialization included; SA can use fewer evaluations when proposals fail. Cached SA evaluations cost less than full GA evaluations, so runtime is also reported. Neither algorithm is granted the other's best solution.

The parameter study varies population (30/60/120), crossover (0.6/0.9/1.0) and mutation (0.01/0.05/0.15) at fixed budget on the 30-task case. This is one-factor-at-a-time evidence and cannot identify all parameter interactions. Cases labeled after literature scales remain synthetic and must not be presented as published benchmarks.

## Recording outline

Record your screen and narrate: introduce the UAV/UGV problem; explain the chromosome and constraints; show ga.py selection/crossover/mutation; run the tests; run the study; show route figures and comparison.png; discuss feasibility, mission time, energy and runtime using RESULTS.md; show parameter results; explain limitations and conclusions. This outline is not a recorded video.

## Submission status

GA, repeated study runner, exact validation and generated results are supplied. REPORT.html is a readable working report. A new measured-results chapter is inserted into the existing LaTeX template by build_ga_report.py. Review the combined report before submission: the inherited chapters still describe Milestone 3. The report includes the existing figures and the new chapter figures under Figures/milestone4. A final compiled report and narrated screen recording remain separate deliverables.

After the study, run `python plot_ga_convergence.py`, `python finish_study.py`, `python build_ga_report.py` and `python build_readable_report.py` to refresh the remaining figures and reports. Run finish_study.py only once per fresh study output because it appends its discussion to RESULTS.md.
