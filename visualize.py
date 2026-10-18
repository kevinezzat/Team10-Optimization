"""Headless report figures with fixed vehicle colors and explicit physical units."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fitness import evaluate
from sa import encode_routes


def save(fig, destination):
    """Save a high-resolution figure and release its memory."""
    fig.savefig(destination, dpi=180, bbox_inches='tight')
    plt.close(fig)


def spread_labels(fig, labels):
    """Separate task-id boxes deterministically without extra plotting packages."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    offsets = [(3, 4), (3, -9), (-10, 4), (-10, -9)]
    offsets += [(dx, dy) for radius in (15, 22, 30, 40)
                for dx, dy in ((radius, 4), (-radius, 4), (3, radius), (3, -radius))]
    for label in labels:
        for offset in offsets:
            label.set_position(offset)
            box = label.get_window_extent(renderer).expanded(1.15, 1.15)
            if not any(box.overlaps(other) for other in occupied):
                break
        occupied.append(box)


def plot_routes(inst, result, destination):
    """Draw route geometry, travel/service timelines, and resource utilization."""
    fig = plt.figure(figsize=(16, 11), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
    ax = fig.add_subplot(grid[:, 0])
    timeline = fig.add_subplot(grid[0, 1])
    bars = fig.add_subplot(grid[1, 1])
    for kind, marker, name in [(0, 'o', 'Any'), (1, 's', 'UGV only'), (2, '^', 'UAV only')]:
        ids = np.flatnonzero(inst.ttype == kind)+1
        ax.scatter(inst.coords[ids, 0], inst.coords[ids, 1], marker=marker,
                   c='lightgray', edgecolors='black', label=name, zorder=3)
    ax.scatter(*inst.coords[0], marker='*', s=200, color='gold', edgecolors='black', label='Depot', zorder=5)
    task_labels = []
    for task in range(1, inst.n+1):
        task_labels.append(ax.annotate(str(task), inst.coords[task], xytext=(3, 4),
                           textcoords='offset points', fontsize=6))
    labels = []
    for k, route in enumerate(result.routes):
        color = plt.cm.tab10(k % 10)
        name = f'{"UAV" if inst.is_uav(k) else "UGV"} {k+1}'
        labels.append(name)
        r = result.details['per_vehicle'][k]
        points = inst.coords[[0]+route+[0]]
        ax.plot(points[:, 0], points[:, 1], '--' if inst.is_uav(k) else '-', color=color,
                label=f'{name}: {r["time"]:.0f}s, {r["energy"]:.0f}kJ, {r["load"]:.1f}kg')
        elapsed = 0.
        for a, b in zip([0]+route, route+[0]):
            travel = inst.dist(k)[a, b]/inst.speed[k]
            timeline.barh(k, travel, left=elapsed, color=color, height=.65)
            elapsed += travel
            if b:
                service = inst.service[b-1]
                timeline.barh(k, service, left=elapsed, color=color, hatch='///', alpha=.4, height=.65)
                elapsed += service
    timeline.axvline(inst.t_max, color='red', linestyle=':', label='Deadline')
    timeline.set(yticks=range(inst.m), yticklabels=labels, xlabel='Time [s]', title='Travel (solid) and service (hatched)')
    timeline.legend(fontsize=8)
    indices = np.arange(inst.m)
    bars.bar(indices-.18, [r['energy']/b for r, b in zip(result.details['per_vehicle'], inst.battery)], .36, label='Battery used / capacity')
    bars.bar(indices+.18, [r['load']/q for r, q in zip(result.details['per_vehicle'], inst.payload)], .36, label='Payload / capacity')
    bars.axhline(1, color='red', linestyle=':')
    bars.set(xticks=indices, xticklabels=labels, ylabel='Fraction of capacity', title='Resource utilization')
    bars.tick_params(axis='x', rotation=45)
    bars.legend(fontsize=8)
    ax.set(xlabel='x [m]', ylabel='y [m]', title=f'{inst.n} tasks | makespan {result.f1:.1f}s | energy {result.f2:.1f}kJ\nViolation {result.violation:.3g}')
    ax.set_aspect('equal')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.08), fontsize=7, ncol=2)
    spread_labels(fig, task_labels)
    save(fig, destination)


def plot_convergence(results, destination, log_y=False):
    """Show current penalized costs and feasible incumbent mean +/- one SD."""
    results = results if isinstance(results, list) else [results]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    histories = np.array([r.best_history for r in results])
    finite = np.isfinite(histories)
    count = finite.sum(axis=0)
    mean = np.divide(np.nansum(histories, axis=0), count, out=np.full(histories.shape[1], np.nan), where=count > 0)
    variance = np.divide(np.nansum((histories-mean)**2, axis=0), count, out=np.zeros_like(mean), where=count > 0)
    sd = np.sqrt(variance)
    x = np.arange(len(mean))
    ax.plot(x, mean, label='Best feasible F (mean)')
    ax.fill_between(x, mean-sd, mean+sd, alpha=.2, label='Across-seed SD')
    ax.plot(results[0].current_history, alpha=.35, label='Current penalized cost (first seed)')
    ax.set(xlabel='Iteration', ylabel='Cost', title='Convergence')
    if log_y:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, destination)


def diagnostics(result, directory):
    """Save schedule, acceptance, and operator counts for a search result."""
    directory = Path(directory)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout='constrained')
    axes[0].plot(result.temperature_history)
    axes[0].set(ylabel='Temperature', title='Cooling and windowed acceptance')
    axes[1].plot(result.acceptance_history)
    axes[1].set(xlabel='Iteration', ylabel='Acceptance fraction', ylim=(0, 1))
    save(fig, directory/'temperature_acceptance.png')
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    names = list(result.operator_stats)
    for j, key in enumerate(('attempt', 'accept', 'improve')):
        ax.bar(np.arange(len(names))+(j-1)*.25, [result.operator_stats[n][key] for n in names], .25, label=key)
    ax.set(xticks=np.arange(len(names)), xticklabels=names, ylabel='Count', title='Neighborhood statistics')
    ax.legend()
    save(fig, directory/'operators.png')


def snapshots(inst, result, directory):
    """Save four intermediate incumbent route frames and the final frame."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for iteration, routes in result.snapshots:
        plot_routes(inst, evaluate(encode_routes(routes, inst), inst), directory/f'iteration_{iteration:06d}.png')
