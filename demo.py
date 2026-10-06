"""
demo.py
=======
Shows how the problem formulation is used:
  1. generate an instance
  2. evaluate ONE random decision vector  (what the TAs will ask to see)
  3. run a pure random-search baseline and plot the best routes and the
     (makespan, energy) trade-off of the feasible solutions found.

Run:  python demo.py            (default: n=30, 2 UGVs, 3 UAVs)
      python demo.py 60 3 4     (n=60 tasks, 3 UGVs, 4 UAVs)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # save figures to file (no window needed)
import matplotlib.pyplot as plt

from instance import generate_instance, UGV_ONLY, UAV_ONLY
from fitness import evaluate, fitness


def plot_routes(inst, result, fname):
    fig, ax = plt.subplots(figsize=(7, 7))
    marker = {0: "o", UGV_ONLY: "s", UAV_ONLY: "^"}
    label = {0: "Any", UGV_ONLY: "UGV-only", UAV_ONLY: "UAV-only"}
    for t in (0, UGV_ONLY, UAV_ONLY):
        idx = np.where(inst.ttype == t)[0] + 1
        ax.scatter(inst.coords[idx, 0], inst.coords[idx, 1], marker=marker[t],
                   c="gray", edgecolors="k", s=45, label=f"Task: {label[t]}")
    ax.scatter(*inst.coords[0], marker="*", s=300, c="gold",
               edgecolors="k", label="Depot", zorder=5)

    for k, route in enumerate(result.routes):
        if not route:
            continue
        pts = inst.coords[np.r_[0, route, 0]]
        uav = inst.is_uav(k)
        ax.plot(pts[:, 0], pts[:, 1], "--" if uav else "-",
                color=plt.cm.tab10(k), lw=1.6,
                label=f"{'UAV' if uav else 'UGV'} {k - inst.m_g + 1 if uav else k + 1}")
    ax.set_title(f"Best random-search solution: makespan = {result.f1:.0f} s, "
                 f"energy = {result.f2:.0f} kJ")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def main():
    n, m_g, m_a = (int(a) for a in sys.argv[1:4]) if len(sys.argv) >= 4 else (30, 2, 3)
    inst = generate_instance(n=n, m_g=m_g, m_a=m_a, seed=0)
    rng = np.random.default_rng(1)

    # ---- 1) evaluate one random decision vector ----
    x = rng.random(2 * inst.n)
    r = evaluate(x, inst)
    print(f"Instance: {inst.n} tasks, {inst.m_g} UGVs, {inst.m_a} UAVs")
    print("--- one random solution ---")
    print(f"makespan f1 = {r.f1:.1f} s | energy f2 = {r.f2:.1f} kJ | "
          f"violation = {r.violation:.3f}")
    print(f"penalised fitness = {fitness(x, inst)}")

    # ---- 2) random-search baseline ----
    N = 20000
    feas, best = [], None
    for _ in range(N):
        x = rng.random(2 * inst.n)
        r = evaluate(x, inst)
        if r.violation == 0:
            feas.append((r.f1, r.f2))
            if best is None or r.f1 < best.f1:
                best = r
    print(f"\n--- random search ({N} samples) ---")
    print(f"feasible solutions found: {len(feas)} ({100 * len(feas) / N:.2f}%)")
    if best is None:
        print("no feasible solution found (this is expected for large n with "
              "pure random search -> the metaheuristic must handle it)")
        return
    print(f"best makespan: {best.f1:.1f} s (energy {best.f2:.1f} kJ)")
    plot_routes(inst, best, "routes.png")

    feas = np.array(feas)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(feas[:, 0], feas[:, 1], s=6, alpha=0.4)
    ax.set_xlabel("f1: makespan [s]"); ax.set_ylabel("f2: total energy [kJ]")
    ax.set_title("Feasible random solutions (objective space)")
    fig.tight_layout(); fig.savefig("objective_space.png", dpi=150)
    print("saved routes.png and objective_space.png")


if __name__ == "__main__":
    main()
