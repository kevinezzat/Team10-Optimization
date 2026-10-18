"""Audit completed experiment artifacts, feasible routes and image contents."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
from experiments import write_json
from fitness import combined_objective, evaluate
from instance import Instance
from sa import SAConfig, encode_routes
from parameter_study import settings
from run_cases import CASES


def main():
    """Fail loudly on missing full-run artifacts or inconsistent recorded solutions."""
    required = [Path('results/cases/all_cases_summary.csv'),
                Path('results/cases/report_table.md'), Path('results/cases/runtime_fit.json'),
                Path('results/validation/exact_gaps.csv'), Path('results/validation/baseline_summary.csv'),
                Path('results/validation/paired_comparisons.json'), Path('results/validation/cvrp.json'),
                Path('results/validation/speed.csv'), Path('results/parameters/findings.md'),
                Path('results/parameters/pareto.png')]
    checked_cases = []
    for name in CASES:
        directory = Path('results/cases')/name
        required += [directory/file for file in ('summary.csv', 'runs.csv', 'parameters.csv',
                     'metrics.json', 'routes.png', 'convergence.png', 'temperature_acceptance.png', 'operators.png')]
        data = json.loads((directory/'metrics.json').read_text())
        fields = data['instance'].copy()
        fields.pop('instance_seed')
        for key, value in fields.items():
            if isinstance(value, list):
                fields[key] = np.asarray(value)
        inst = Instance(**fields)
        result = evaluate(encode_routes(data['best_routes'], inst), inst)
        best_run = next(r for r in data['runs'] if r['seed'] == data['best_seed'])
        assert abs(combined_objective(result, inst, data['config']['alpha'])-best_run['F']) <= 1e-9
        assert abs(result.violation-best_run['violation']) <= 1e-9
        if best_run['feasible']:
            assert result.violation <= 1e-9
        assert len(data['runs']) >= 10
        checked_cases.append(dict(case=name, feasible=best_run['feasible'], violation=result.violation))
    for name in CASES:
        budgets = []
        for algorithm in ('sa', 'random', 'greedy', 'hill'):
            path = Path('results/validation')/f'{name}_{algorithm}.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) >= 20, f'incomplete baseline: {path}'
            budgets.append([int(row['evaluations']) for row in rows])
            required.append(path)
        assert all(budget == budgets[0] for budget in budgets), f'budget mismatch: {name}'
    for n in (6, 7, 8, 9):
        path = Path('results/validation')/f'exact_n{n}_runs.csv'
        with path.open(newline='') as stream:
            assert len(list(csv.DictReader(stream))) >= 20
        required.append(path)
    for factor, value, _ in settings(SAConfig(construction_restarts=10)):
        slug = value.replace(',', '_').replace('=', '-')
        path = Path('results/parameters')/f'{factor}_{slug}_runs.csv'
        with path.open(newline='') as stream:
            assert len(list(csv.DictReader(stream))) >= 10, f'incomplete sweep: {path}'
        required.extend([path, Path('results/parameters')/f'{factor}.csv',
                         Path('results/parameters')/f'{factor}.png'])
    for path in required:
        assert path.is_file() and path.stat().st_size > 0, f'missing/empty: {path}'
    images = []
    for directory in ('outputs_n30', 'outputs_sa_n30', 'results'):
        for path in Path(directory).rglob('*.png'):
            pixels = mpimg.imread(path)
            assert min(pixels.shape[:2]) >= 400, f'too small: {path}'
            assert float(pixels[..., :3].std()) > .02, f'blank: {path}'
            images.append(dict(path=str(path), width=pixels.shape[1], height=pixels.shape[0]))
    write_json('results/verification.json', dict(cases=checked_cases, images=images,
               required_files=len(required), note='Image dimensions/content checks do not prove absence of every label overlap.'))
    print(f'Verified {len(checked_cases)} case solutions, {len(required)} required files and {len(images)} nonblank figures.')


if __name__ == '__main__':
    main()
