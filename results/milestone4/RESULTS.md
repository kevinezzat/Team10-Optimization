# Milestone 4 measured results

5 seeds per configuration; evaluation budget 3000. Instance seed 0. Alpha 0.5.

| Case | Algorithm | Feasible | Mean F | Mean time (s) |
|---|---|---:|---:|---:|
| battery75 | GA | 0% | None | 6.228 |
| battery75 | SA | 0% | None | 6.917 |
| crossover_0.6 | GA | 100% | 0.22941751091549323 | 1.432 |
| crossover_0.9 | GA | 100% | 0.23081343601559023 | 1.481 |
| crossover_1.0 | GA | 100% | 0.23137848024648724 | 1.537 |
| exact6 | GA | 100% | 0.22516961728926738 | 1.674 |
| exact6 | SA | 100% | 0.22516961728926738 | 3.944 |
| exact7 | GA | 100% | 0.2821297454042614 | 1.627 |
| exact7 | SA | 100% | 0.2821297454042614 | 2.932 |
| gao100 | GA | 100% | 0.5875840944631816 | 3.986 |
| gao100 | SA | 100% | 0.44535965591506355 | 4.484 |
| gao50 | GA | 100% | 0.5901681661752564 | 4.857 |
| gao50 | SA | 100% | 0.5797902164003522 | 5.051 |
| mutation_0.01 | GA | 100% | 0.22457955210906047 | 1.381 |
| mutation_0.05 | GA | 100% | 0.23081343601559023 | 1.594 |
| mutation_0.15 | GA | 100% | 0.23137848024648724 | 3.038 |
| population_120 | GA | 100% | 0.2247949532808752 | 1.675 |
| population_30 | GA | 100% | 0.227849258500066 | 1.258 |
| population_60 | GA | 100% | 0.23081343601559023 | 1.536 |
| reference30 | GA | 100% | 0.23081343601559023 | 3.837 |
| reference30 | SA | 100% | 0.2145655857859896 | 4.039 |
| restricted90 | GA | 100% | 0.6014147056228437 | 4.426 |
| restricted90 | SA | 100% | 0.5597059749786222 | 4.089 |
| small10 | GA | 100% | 0.318378088494614 | 3.909 |
| small10 | SA | 100% | 0.318378088494614 | 34.013 |

Objectives, makespan and energy are summarized over feasible runs only. Infeasible runs remain in runs.csv with their violation.
GA uses its full evaluation budget. SA has the same upper bound but may use fewer evaluations when proposal generation fails. Runtime includes construction. Evaluation counts are not identical costs: SA caches unchanged routes.
These are synthetic instances, not reproduced literature datasets. Exact small-instance optima provide validation. No universal superiority claim is supported. Parameter studies hold the evaluation budget fixed; larger populations therefore receive fewer generations.

## Interpretation of measured results

small10: mean feasible GA objective 0.31838, SA 0.31838; GA is 0.00% lower. Feasibility rates: GA 100%, SA 100%.

reference30: mean feasible GA objective 0.23081, SA 0.21457; GA is 7.57% higher. Feasibility rates: GA 100%, SA 100%.

gao50: mean feasible GA objective 0.59017, SA 0.57979; GA is 1.79% higher. Feasibility rates: GA 100%, SA 100%.

battery75: GA feasibility 0%, SA 0%. No objective comparison is made without feasible solutions; failure to find a feasible solution is not a proof of infeasibility.

restricted90: mean feasible GA objective 0.60141, SA 0.55971; GA is 7.45% higher. Feasibility rates: GA 100%, SA 100%.

gao100: mean feasible GA objective 0.58758, SA 0.44536; GA is 31.93% higher. Feasibility rates: GA 100%, SA 100%.

Among tested population values, 120 had the lowest mean objective (0.22479) on the 30-task instance. This is descriptive evidence from five seeds, not an independently validated tuning recommendation.

Among tested crossover values, 0.6 had the lowest mean objective (0.22942) on the 30-task instance. This is descriptive evidence from five seeds, not an independently validated tuning recommendation.

Among tested mutation values, 0.01 had the lowest mean objective (0.22458) on the 30-task instance. This is descriptive evidence from five seeds, not an independently validated tuning recommendation.
