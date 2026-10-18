# Report alignment checks

No LaTeX report is present in this checkout. The following are code facts and
items to reconcile with the report, not claims about unseen report text.

- Constraints use **nonnegative violation**, with zero feasible. If the report
  defines `g_j <= 0`, distinguish residual functions from these clipped penalties.
- Capability violation is incompatible-task count divided by n. Payload, energy,
  and deadline violations sum normalized per-vehicle excesses, not averages.
- The battery normalization is `sum(B_k)` across vehicles, including multiplicity
  of vehicle types. UAV and UGV type batteries are separate; units are kJ.
- Deadline includes return to depot and task service time, with no release times,
  time windows, rendezvous, charging, or inter-vehicle synchronization.
- UGV distance is Euclidean distance times a detour factor. No unicycle state,
  heading, steering, collision avoidance or kinematics equation is simulated.
- Legacy `fitness()` multiplies each objective by `1 + penalty*violation` for
  GA/PSO compatibility. SA uses additive `F + lambda*violation` instead.
- SA stores routes internally, but public inputs/results retain the exact
  allocation and ascending sequence random-key encoding.
- Generated capacity scaling is a heuristic, not proof of feasible instances.
- A finite positive T_max and finite positive summed battery are required for
  report normalization. The demo's unbounded deadline is for separate objectives.
- The CVRP fixture is synthetic and exactly solved; it is **not** Augerat/E-n22
  data and cannot support a published-benchmark performance claim.
- Chen/Gao/Wang names in case labels describe sizes supplied in the assignment
  prompt, not a verified bibliographic or numerical comparison. Verify citations
  and fleet details against the actual literature review before submission.
- Statistical wins against random search use feasibility-first ranking; report
  objective differences only for pairs in which both algorithms are feasible.
- Cross-case normalized F changes with deadline and battery denominators, so its
  scalability curve is descriptive and does not compare identical objectives.
