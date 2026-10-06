"""
instance.py
===========
Problem data for the Heterogeneous UAV-UGV Cooperative Task Allocation and
Routing problem (MCTR 1021 - Milestone 2).

An `Instance` stores everything that is FIXED during optimisation
(task locations, demands, fleet parameters).  The optimiser only changes the
decision vector x (see fitness.py).

Units
-----
distance [m] | time [s] | speed [m/s] | payload [kg] | energy [kJ]

Vehicle indexing
----------------
Vehicles are numbered 0 .. m-1 with m = m_g + m_a
    0 .. m_g-1      -> UGVs (ground)
    m_g .. m-1      -> UAVs (aerial)

Task types
----------
    0 : ANY       - can be served by a UAV or a UGV
    1 : UGV_ONLY  - e.g. heavy / ground-contact task, a UAV cannot do it
    2 : UAV_ONLY  - e.g. location unreachable on the ground
"""
from dataclasses import dataclass
import numpy as np

ANY, UGV_ONLY, UAV_ONLY = 0, 1, 2


@dataclass
class Instance:
    # ---- tasks (index 0 is reserved for the depot in the matrices) ----
    n: int                  # number of tasks
    coords: np.ndarray      # (n+1, 2) depot at row 0, task i at row i (1..n)
    demand: np.ndarray      # (n,)  payload needed by task i   [kg]
    service: np.ndarray     # (n,)  service/ dwell time at task i [s]
    ttype: np.ndarray       # (n,)  task type (ANY / UGV_ONLY / UAV_ONLY)
    # ---- fleet (arrays of length m) ----
    m_g: int                # number of UGVs
    m_a: int                # number of UAVs
    speed: np.ndarray       # [m/s]
    payload: np.ndarray     # max payload carried per mission Q_k [kg]
    e_rate: np.ndarray      # energy used per metre travelled [kJ/m]
    battery: np.ndarray     # energy budget B_k [kJ]
    # ---- mission ----
    t_max: float            # mission deadline [s]  (np.inf = no deadline)
    # ---- distance matrices (n+1, n+1) ----
    D_ugv: np.ndarray       # ground distance (Euclidean x detour factor)
    D_uav: np.ndarray       # aerial distance (straight line)

    @property
    def m(self) -> int:
        return self.m_g + self.m_a

    def is_uav(self, k: int) -> bool:
        return k >= self.m_g

    def dist(self, k: int) -> np.ndarray:
        """Distance matrix that vehicle k travels on."""
        return self.D_uav if self.is_uav(k) else self.D_ugv


def generate_instance(n=30, m_g=2, m_a=3, seed=0, area=1000.0,
                      p_ugv_only=0.2, p_uav_only=0.2,
                      ugv_detour=1.2, t_max=np.inf) -> Instance:
    """
    Random instance generator (valid for ANY problem size n, m_g, m_a).

    Capacities / batteries are scaled with the problem size so that
    feasible solutions exist for every (n, m_g, m_a).

    Parameters
    ----------
    n          : number of tasks
    m_g, m_a   : number of UGVs / UAVs
    seed       : RNG seed (reproducible instances for benchmarking)
    area       : side length of the square workspace [m]
    p_ugv_only : probability that a task can only be done by a UGV
    p_uav_only : probability that a task can only be done by a UAV
    ugv_detour : factor (>=1) modelling obstacle detours on the ground
                 (Wang et al. 2024 use 1.2)
    t_max      : mission deadline [s]
    """
    rng = np.random.default_rng(seed)
    m = m_g + m_a

    # ---- geometry: depot in the centre, tasks uniformly random ----
    depot = np.array([[area / 2, area / 2]])
    tasks = rng.uniform(0, area, size=(n, 2))
    coords = np.vstack([depot, tasks])
    diff = coords[:, None, :] - coords[None, :, :]
    euclid = np.sqrt((diff ** 2).sum(-1))

    # ---- task attributes ----
    demand = rng.uniform(0.5, 3.0, size=n)            # kg
    service = rng.uniform(10.0, 30.0, size=n)         # s
    ttype = rng.choice([ANY, UGV_ONLY, UAV_ONLY], size=n,
                       p=[1 - p_ugv_only - p_uav_only, p_ugv_only, p_uav_only])

    # ---- fleet parameters ----
    speed = np.r_[np.full(m_g, 2.0), np.full(m_a, 10.0)]          # m/s
    e_rate = np.r_[np.full(m_g, 0.08), np.full(m_a, 0.03)]        # kJ/m

    # Reference tour length per vehicle (Beardwood-Halton-Hammersley estimate)
    share = max(n / m, 1.0)
    L_ref = 0.7124 * np.sqrt(share * area ** 2) + area          # + depot legs
    # UAV range limited (small battery), UGV range generous
    battery = np.r_[np.full(m_g, 3.0 * L_ref * ugv_detour * 0.08),
                    np.full(m_a, 2.5 * L_ref * 0.03)]

    # Payload: UGVs carry more than UAVs; UAVs can always carry any single task
    D_total = demand.sum()
    Q_uav = max(1.5 * D_total / m, 2.0 * demand.max())
    Q_ugv = max(3.0 * D_total / m, 2.0 * demand.max())
    payload = np.r_[np.full(m_g, Q_ugv), np.full(m_a, Q_uav)]

    return Instance(n=n, coords=coords, demand=demand, service=service,
                    ttype=ttype, m_g=m_g, m_a=m_a, speed=speed,
                    payload=payload, e_rate=e_rate, battery=battery,
                    t_max=t_max, D_ugv=ugv_detour * euclid, D_uav=euclid)
