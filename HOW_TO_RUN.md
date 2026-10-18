# Reproducing Milestone 3

Use Python 3.10+ from the repository root. All figures use Agg; no GUI or GPU is needed.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Use `.venv/bin/python` on Linux/macOS. After activation, run:

```bash
python -m unittest discover -s tests -v
python demo.py
python run_sa.py
python run_cases.py --workers 2
python -m validation --workers 3
python -m validation.cvrp
python -m validation.speed
python parameter_study.py --workers 3
python verify_outputs.py
```

On this machine the Windows Store Python alias was inaccessible, so actual commands
used `.venv/Scripts/python.exe` instead of `python`.

| Command | Full defaults | Output |
|---|---|---|
| tests | Includes 3000 random moves | Console; initially about 7 seconds |
| demo | 20000 samples, seed 1 | Root route/objective figures when feasible samples exist |
| run_sa | 30 tasks, 12000 iterations, 100 starts, seed 1 | outputs_sa_n30/, including PNG frames |
| run_cases | Seven cases, 10 seeds, 12000 iterations, 10 starts | results/cases/: raw runs, parameters, metrics, figures and report table |
| validation | n=6-9 exact comparison; seven baseline cases; 20 seeds, 2000 iterations | results/validation/: exact gaps, baseline summaries and paired comparisons |
| validation.cvrp | Synthetic 8-customer CVRP, 20 seeds | results/validation/cvrp.json |
| validation.speed | 3000 updates, 3 repetitions per size | results/validation/speed.csv |
| parameter_study | 10 seeds per setting, iterations up to 60000 | results/parameters/: CSVs, figures, Pareto plot, findings.md |

The original single run took 14.32 seconds. Initial upgraded runs took roughly
9-15 seconds depending on size and concurrent workload. Full repeated studies
take many minutes to hours. Greedy baseline construction dominates validation
cost. Use saved wall times for actual measurements; estimates are not guarantees.
Parameter studies support `--resume`; reuse checkpoints only with the same source
and inputs. Configuration is checked but source-version migrations are not.
Case studies run sequentially without `--workers`, with a sequential fallback if
process startup fails. All process entry points use the standard main guard.

Development smoke tests (not report settings):

```bash
python run_cases.py --cases small10 --seeds 2 --iterations 100 --restarts 1 --output results/smoke
python -m validation --seeds 2 --iterations 100 --skip-baselines --output results/smoke
```

Use a separate `--output` for exploratory runs. See each CLI's `--help` for controls.
A customized single run:

```bash
python run_sa.py --tasks 50 --ugvs 3 --uavs 3 --deadline 1200 --sa-seed 7 --iterations 30000 --target-acceptance 0.8 --penalty 100 --output outputs_custom
```

Seeds are recorded in metadata. Parameter sweeps use one fixed 30-task instance,
so they do not establish universal defaults. Alpha changes F itself: compare
physical time/energy across alpha. Cross-case F also uses different denominators.

CVRP data are original synthetic data with an exhaustive optimum, not published
Augerat/E-n benchmark data. Chen/Gao/Wang labels match sizes from the assignment
prompt only; verify citations against the actual literature review.
