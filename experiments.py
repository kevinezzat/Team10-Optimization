"""Shared reproducible experiment records and descriptive statistics."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import numpy as np
from fitness import combined_objective, evaluate
from sa import SAConfig, simulated_annealing


def write_json(path, data):
    """Write strict JSON, converting numpy values explicitly."""
    def convert(value):
        """Convert array/scalar values without hiding nonfinite measurements."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    """Write homogeneous rows with their explicit field names."""
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def instance_metadata(inst, seed):
    """Capture all input data, not only generator arguments, for exact replay."""
    return dict(asdict(inst), instance_seed=seed)


def run_one(job):
    """Top-level picklable worker, returning independently checked measurements."""
    inst, config, seed = job
    result = simulated_annealing(inst, config, seed)
    checked = evaluate(result.best_x, inst)
    assert abs(checked.violation-result.best_result.violation) <= 1e-9
    if result.feasible:
        assert checked.violation <= 1e-9
        if result.initial_result.violation <= 1e-9:
            assert combined_objective(checked, inst, config.alpha) <= combined_objective(result.initial_result, inst, config.alpha)+1e-9
    row = dict(seed=seed, F=combined_objective(checked, inst, config.alpha),
               makespan=checked.f1, energy=checked.f2, runtime=result.wall_time,
               feasible=result.feasible, violation=checked.violation,
               evaluations=result.evaluations)
    return row, result


def summarize(rows):
    """Summarize feasible objectives separately from all-run runtime and work."""
    output = dict(runs=len(rows), feasibility_rate=float(np.mean([r['feasible'] for r in rows])))
    for key in ('F', 'makespan', 'energy', 'runtime', 'evaluations'):
        values = [r[key] for r in rows if r['feasible'] or key in ('runtime', 'evaluations')]
        for stat in ('best', 'mean', 'std', 'median'):
            output[f'{key}_{stat}'] = (None if not values else float(
                np.min(values) if stat == 'best' else np.mean(values) if stat == 'mean'
                else np.std(values, ddof=1) if stat == 'std' and len(values)>1
                else 0. if stat == 'std' else np.median(values)))
    return output
