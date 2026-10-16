"""
fitness.py
==========
Decision-variable decoding and fitness (objective + constraint) evaluation for
the Heterogeneous UAV-UGV Cooperative Task Allocation and Routing problem.

DECISION VARIABLES  (what the metaheuristic optimises)
------------------------------------------------------
A single real vector  x in [0, 1]^(2n)  ("random-key" encoding):

    x[0:n]   -> allocation keys  : task i is assigned to vehicle
                                   a_i = min(floor(x_i * m), m-1)
    x[n:2n]  -> sequence keys    : tasks of the same vehicle are visited in
                                   ascending order of their key

Equivalent mathematical decision variables:
    a_i  in {0..m-1}   vehicle that serves task i        (ALLOCATION)
    pi_k = ordered list of tasks of vehicle k            (ROUTING / SEQUENCE)

Every x in [0,1]^(2n) decodes to a valid allocation + sequence, so every task
is served exactly once and every route starts / ends at the depot BY
CONSTRUCTION.  Any continuous metaheuristic (GA, PSO, DE, ...) can therefore
be used directly.

OBJECTIVES (both minimised)
---------------------------
    f1 = makespan            = max_k T_k
    f2 = total energy        = sum_k E_k

CONSTRAINTS  (violation measure, 0 = feasible)
----------------------------------------------
    g1  capability : UGV-only tasks not on UAVs, UAV-only tasks not on UGVs
    g2  payload    : sum_{i in route k} q_i <= Q_k
    g3  energy     : E_k <= B_k
    g4  deadline   : T_k <= T_max
"""
from collections import namedtuple
import numpy as np
from instance import Instance, UGV_ONLY, UAV_ONLY

# f1, f2 : objective values | violation : total normalised constraint violation
Result = namedtuple("Result", ["f1", "f2", "violation", "routes", "details"])


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def decode(x: np.ndarray, inst: Instance):
    """
    Decode a decision vector into one ordered route per vehicle.

    Returns
    -------
    routes : list of length m; routes[k] is the list of task ids (1..n) that
             vehicle k visits, in visiting order (depot = 0 is implicit at
             the start and end of every route).
    """
    n, m = inst.n, inst.m
    x = np.asarray(x, dtype=float)
    assert x.size == 2 * n, f"x must have length 2n = {2 * n}"

    vehicle = np.minimum((x[:n] * m).astype(int), m - 1)     # a_i
    order = np.argsort(x[n:], kind="stable")                 # global ordering

    routes = [[] for _ in range(m)]
    for i in order:                      # tasks in ascending sequence-key order
        routes[vehicle[i]].append(int(i) + 1)   # +1 : task ids start at 1
    return routes


# --------------------------------------------------------------------------
# Single-route evaluation
# --------------------------------------------------------------------------
def route_metrics(k: int, route, inst: Instance):
    """Distance, time, energy and payload of vehicle k following `route`."""
    if len(route) == 0:
        return dict(dist=0.0, time=0.0, energy=0.0, load=0.0)

    seq = np.r_[0, route, 0]                         # depot -> tasks -> depot
    dist = inst.dist(k)[seq[:-1], seq[1:]].sum()     # total distance  [m]
    t_service = inst.service[np.array(route) - 1].sum()
    time = dist / inst.speed[k] + t_service          # T_k  [s]
    energy = inst.e_rate[k] * dist                   # E_k  [kJ]
    load = inst.demand[np.array(route) - 1].sum()    # sum q_i [kg]
    return dict(dist=dist, time=time, energy=energy, load=load)


# --------------------------------------------------------------------------
# Full evaluation
# --------------------------------------------------------------------------
def evaluate(x: np.ndarray, inst: Instance) -> Result:
    """
    Compute objectives f1, f2 and the total constraint violation for x.

    violation == 0  <=>  x decodes to a feasible solution.
    """
    routes = decode(x, inst)
    m = inst.m

    times, energies, per_vehicle = np.zeros(m), np.zeros(m), []
    g_capability = g_payload = g_energy = g_deadline = 0.0

    for k in range(m):
        r = route_metrics(k, routes[k], inst)
        per_vehicle.append(r)
        times[k], energies[k] = r["time"], r["energy"]

        # g1: capability (count of tasks given to a vehicle that cannot do it)
        for i in routes[k]:
            t = inst.ttype[i - 1]
            if (t == UGV_ONLY and inst.is_uav(k)) or \
               (t == UAV_ONLY and not inst.is_uav(k)):
                g_capability += 1.0 / inst.n
        # g2: payload   max(0, load - Q_k) / Q_k
        g_payload += max(0.0, r["load"] - inst.payload[k]) / inst.payload[k]
        # g3: energy    max(0, E_k - B_k) / B_k
        g_energy += max(0.0, r["energy"] - inst.battery[k]) / inst.battery[k]
        # g4: deadline  max(0, T_k - T_max) / T_max  (skipped if no deadline)
        if np.isfinite(inst.t_max):
            g_deadline += max(0.0, r["time"] - inst.t_max) / inst.t_max

    f1 = float(times.max())          # makespan
    f2 = float(energies.sum())       # total energy
    violation = g_capability + g_payload + g_energy + g_deadline

    details = dict(per_vehicle=per_vehicle, g_capability=g_capability,
                   g_payload=g_payload, g_energy=g_energy,
                   g_deadline=g_deadline)
    return Result(f1, f2, float(violation), routes, details)


def fitness(x: np.ndarray, inst: Instance, penalty: float = 1e4):
    """
    Penalised fitness vector (F1, F2) to be MINIMISED by a metaheuristic:

        F_j = f_j * (1 + penalty * violation)

    Feasible solutions keep their true objective values; infeasible ones are
    pushed far away from the feasible region in proportion to the violation.
    """
    r = evaluate(x, inst)
    scale = 1.0 + penalty * r.violation
    return r.f1 * scale, r.f2 * scale


def weighted_fitness(x, inst, w=(0.5, 0.5), ref=(1.0, 1.0), penalty=1e4):
    """Single-objective version for GA/PSO: weighted sum of normalised f1, f2.
    `ref` are normalisation constants (e.g. best values from a pilot run)."""
    F1, F2 = fitness(x, inst, penalty)
    return w[0] * F1 / ref[0] + w[1] * F2 / ref[1]


def combined_objective(result: Result, inst: Instance, alpha: float = 0.5) -> float:
    """Report's scalar objective: alpha*f1/T_max + (1-alpha)*f2/sum(B_k).

    The mission deadline must be finite: if T_max is infinity, the makespan
    term would silently become zero and the objective would be incorrect.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie between 0 and 1")
    if not np.isfinite(inst.t_max) or inst.t_max <= 0:
        raise ValueError("combined objective requires a positive finite t_max")
    total_battery = float(inst.battery.sum())
    if total_battery <= 0 or not np.isfinite(total_battery):
        raise ValueError("total battery capacity must be positive and finite")
    return float(alpha * result.f1 / inst.t_max
                 + (1.0 - alpha) * result.f2 / total_battery)


def sa_fitness(result: Result, inst: Instance, alpha: float = 0.5,
               penalty: float = 100.0) -> float:
    """Scalar cost to minimize in SA; infeasibility is penalized additively.

    In feasible states this equals the exact combined objective in the report.
    """
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError("penalty must be finite and nonnegative")
    return combined_objective(result, inst, alpha) + penalty * result.violation
