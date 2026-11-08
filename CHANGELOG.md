# Milestone 4 upgrade

- Added a hand-written GA with tournament selection, uniform crossover, mutation, capability repair and elitism.
- Added 125 recorded experiment runs, exact small-case validation, SA comparisons and parameter studies.
- Added route, convergence and comparison figures, working HTML report, and a GA chapter in the existing LaTeX report.
- Added GA tests; all 16 tests pass.
- See MILESTONE4_GUIDE.md for reproduction and remaining submission work.

# Milestone 3 upgrade

## Baseline (2026-10-18)

Read the original modules, tests, run guide, dependency/Git files, stored metrics
and figure folders before implementation. No LaTeX report or AGENTS.md was found.
Original six tests passed. Reproduced default SA: initial F **0.2269084478**,
best F **0.1790990501**, 3960/12000 accepted (**33%**), violation 0,
wall time **14.3165 s**. Captured reproduction: `baseline_original/`.
Original result directories have been retained.

The Windows Python alias was inaccessible; Python 3.10 lacked dependencies.
Created an ignored `.venv` with NumPy/Matplotlib and their pip-managed transitive
dependencies. Windows sandbox helper failures required approved external execution.

## Changes

- `sa.py`, `sa_engine.py`: frozen validated configuration, calibrated/derived
  cooling, five compatible operators, adaptive selection/penalties, reheating,
  independent chains, route caches, diagnostics and flagged infeasible fallback.
- Vectorized greedy edge insertion; initialization now actually prefers feasible
  candidates. Alpha informs insertion. SeedSequence streams change old trajectories.
- Preserved Instance/decode/evaluate/combined_objective/sa_fitness signatures and
  model formulas. `decode` now raises ValueError for invalid keys. Temperature
  defaults to automatic (`None`). SAResult retains original fields and adds diagnostics.
- `validation/`: exact solver, random/greedy/hill comparisons, paired sign tests
  and bootstrap intervals, synthetic CVRP reduction and evaluation timing replay.
- `run_cases.py`, `parameter_study.py`, `experiments.py`: seeded studies, raw CSV,
  strict JSON, parallel workers, statistical summaries and parameter checkpoints.
- `visualize.py`, `run_sa.py`: headless route/timeline/resource figures, histories,
  acceptance/operator diagnostics and PNG snapshots.
- `tests/`: cache equality after thousands of moves, hand-computed constraints,
  exact solutions, invalid settings and infeasible-return regressions.
- Documentation corrects the unsupported claim that generator capacity scaling
  guarantees feasibility. Report assumptions are listed separately.

## Interpretation

Complete-candidate evaluation counts exclude partial greedy insertion scores;
equal counts do not mean equal CPU work. Runtime is recorded separately. Feasible
objective summaries exclude infeasible runs; paired wins rank feasibility first.
Runtime under concurrent studies is workload-specific. Synthetic CVRP validation
is not a published benchmark claim. See generated results for measured findings;
the existence of a runner is not evidence that its experiment finished.
