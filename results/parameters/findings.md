# Measured parameter findings

One instance (30 tasks, instance seed 0); paired seeds. SD describes run variation, not uncertainty of the mean.
Alpha changes the objective itself: compare makespan and energy, not F across alpha.

- iterations: lowest observed mean F at 60000 (0.174115 ± 0.006814); highest at 2000 (0.206826). These descriptive rankings are not significance tests.
- target_acceptance: lowest observed mean F at 0.2 (0.186621 ± 0.014407); highest at 0.8 (0.193968). These descriptive rankings are not significance tests.
- final_temperature: lowest observed mean F at 1e-07 (0.189556 ± 0.006914); highest at 0.001 (0.209214). These descriptive rankings are not significance tests.
- penalty: lowest observed mean F at 10.0 (0.184965 ± 0.005275); highest at 100.0 (0.193968). These descriptive rankings are not significance tests.
- construction_restarts: lowest observed mean F at 1 (0.188611 ± 0.010961); highest at 10 (0.193968). These descriptive rankings are not significance tests.
- adaptive_operators: lowest observed mean F at True (0.189557 ± 0.008507); highest at False (0.193968). These descriptive rankings are not significance tests.
- reheat: lowest observed mean F at False (0.193968 ± 0.006071); highest at True (0.245779). These descriptive rankings are not significance tests.
- ablation: lowest observed mean F at without_intra_swap (0.183227 ± 0.007683); highest at only_intra_swap (0.244856). These descriptive rankings are not significance tests.
- factorial: lowest observed mean F at p=0.5,lambda=10.0 (0.181836 ± 0.009689); highest at p=0.8,lambda=100.0 (0.193968). These descriptive rankings are not significance tests.

Recommendations should balance the measured runtime and feasibility against F. Keep alpha tied to mission priorities; this single-instance study does not establish universal defaults.
