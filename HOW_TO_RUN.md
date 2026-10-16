# How to Run the Code

## 1) Prerequisites
- Python 3.9+ (recommended)
- `pip`

## 2) Install Dependencies
From the repository root:

```bash
pip install -r requirements.txt
```

## 3) Run the Demo
Default run (30 tasks, 2 UGVs, 3 UAVs):

```bash
python demo.py
```

Custom run (`n m_g m_a`):

```bash
python demo.py 60 3 4
```

## 4) Output Files
After a successful run, the script saves:
- `routes.png`
- `objective_space.png`

## 5) Milestone 3 — Simulated Annealing

The SA implementation **uses the existing** `instance.py` generator and
`fitness.py` decoder / evaluator. No alternate model or dataset is created.
Its optimization target is the updated report's scalar objective:
`F = 0.5 * makespan / T_max + 0.5 * energy / sum(vehicle batteries)`.
Batteries and energy are measured in **kJ**, not Wh. All UAVs share one battery
value and all UGVs share another, as in the report.

Run from the repository root:

```bash
python run_sa.py
```

Customize the reproducible scenario or SA budget:

```bash
python run_sa.py --tasks 30 --ugvs 2 --uavs 3 --instance-seed 0 --sa-seed 1 --deadline 1800 --iterations 12000 --output outputs_sa_n30
```

Outputs: `outputs_sa_n30/metrics.json`, `convergence.png`, `routes.png`.
The saved JSON contains route assignments and the four constraint violations.

The **finite** `--deadline` is essential for the normalization in the report.
The original `demo.py` uses `t_max=inf` for its unconstrained random-search
example; it is preserved unchanged and continues to run as before.

Run tests:

```bash
python -m unittest discover -s tests -v
```

SA uses capability-aware neighborhood moves (relocate, swap, reverse), Metropolis
acceptance, geometric cooling, an additive constraint penalty, and separate
tracking of the best feasible solution. Random seeds allow fair comparisons
with later GA/PSO/ACO/etc. using the same `generate_instance` calls.
