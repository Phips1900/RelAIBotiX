# Real Franka sorting: changed objects and switched boxes

## Scope and selection

This supplementary analysis compares five variation datasets with the three frozen
standard real-Franka sorting baselines. The publication configuration
[`franka.json`](../../../configs/robots/franka.json),
`additive_normalized` exposure model, tenfold component sensitivity perturbation,
and final-Place stopping rule are shared by all conditions.

The existing sorting-transformer predictions use the real-Franka input adapter,
five-frame minimum duration, and `franka-manipulation` transition profile.
Behavioral exposure was recomputed for the variation datasets. Recomputed standard reliability
matches the frozen results.

Only no-pick runs explicitly identified in the experiment notes are excluded. Incorrect
container choices and outside-container placements are retained. Task-success
metadata is not used to select runs. The exclusion numbers below use one-based
recording order, corresponding to canonical keys `demo_000000` through
`demo_000019` (episode number minus one).

| Policy | Variation | Excluded note episodes | Retained |
|---|---|---|---:|
| ACT | New colors/sizes/shapes | 2, 4, 10 | 17/20 |
| pi0.5 | New colors/sizes/shapes | None | 20/20 |
| SmolVLA | New colors/sizes/shapes | 2, 5, 6, 8 | 16/20 |
| pi0.5 | Switched boxes | 7 | 19/20 |
| SmolVLA | Switched boxes | 2, 8 | 18/20 |

All retained variation episodes contain a Place prediction. Each standard baseline
contains one episode without Place, retained in full under the frozen protocol.
ACT has no switched-box dataset in this comparison. The new-object runs pool
yellow-object recordings 1–10 and blue-object recordings 11–20; the switched-box
runs pool green-object recordings 1–10 and red-object recordings 11–20.

## Results

Changes are relative to the same policy's standard sorting condition. MTTF is in
operating hours for repeated execution under the model assumptions.

| Policy | Condition | Runs | Mean time (s) | Failure probability/run | Change | MTTF (h) | Change |
|---|---|---:|---:|---:|---:|---:|---:|
| ACT | Standard | 99 | 31.13 | 6.666e-7 | — | 12,971 | — |
| ACT | New objects | 17 | 30.36 | 6.491e-7 | -2.62% | 12,993 | +0.17% |
| pi0.5 | Standard | 92 | 26.08 | 5.531e-7 | — | 13,101 | — |
| pi0.5 | New objects | 20 | 26.60 | 5.616e-7 | +1.54% | 13,156 | +0.42% |
| pi0.5 | Switched boxes | 19 | 25.71 | 5.423e-7 | -1.95% | 13,172 | +0.54% |
| SmolVLA | Standard | 96 | 28.47 | 6.015e-7 | — | 13,148 | — |
| SmolVLA | New objects | 16 | 28.59 | 6.030e-7 | +0.26% | 13,169 | +0.16% |
| SmolVLA | Switched boxes | 18 | 27.73 | 5.852e-7 | -2.71% | 13,161 | +0.10% |

Joint 2, Joint 4, and Joint 6 remain the first three sensitivity-ranked components
in that order in every condition. Failure probability changes by at most 2.71%,
whereas MTTF changes by at most 0.54%. Run duration changes by -2.61% to +1.97%.
Total joint travel changes by -2.53% to +5.10%. These recordings do not show a
substantial deterioration in modeled hardware reliability under the variations.

## What skill decomposition reveals

Bottom-up and BDD fault-tree probabilities agree. Exact PRISM and Storm execution
verified failure/completion probabilities and repeated-operation MTTF for all five
variation models; the verification records are saved in `solver_verification.json`.

The following are DTMC absorption contributions to total modeled failure
probability. They sum to 100% within each condition.

| pi0.5 | Move | Pick | Carry | Place |
|---|---:|---:|---:|---:|
| Standard | 34.19% | 32.85% | 13.35% | 19.60% |
| New objects | 31.25% | 34.02% | 9.13% | 25.60% |
| Switched boxes | 32.81% | 30.74% | 16.43% | 20.02% |

For pi0.5 with new objects, the absolute Place contribution increases by 32.61%,
while the absolute Carry contribution decreases by 30.58%. Detector-assigned mean
Place duration increases from 5.09 to 6.79 s; Carry duration decreases from 3.47 to
2.46 s. Total failure probability increases by only 1.54% and MTTF by 0.42%.
For switched boxes, pi0.5's absolute Carry contribution increases by 20.68%, while
its total failure probability decreases by 1.95%.

ACT with new objects similarly has a 13.68% increase in its Carry contribution
despite a 2.62% decrease in total failure probability. These observations illustrate
how compensating changes across skills can be hidden in the aggregate result.

These are observations using predicted skill labels, not independent confirmation
of a change in physical phase boundaries. The Carry/Place redistribution depends
on the detector's phase definitions. The small, unpaired variation sets do not
establish statistical equivalence or a causal effect of object appearance.

## Interpretation and task success

Task-success accuracy decreased substantially in some variations. Wrong-container
placements can involve similar joint loading and motion to correct placements, so
reduced task success can coexist with similar modeled hardware reliability.
Task-success accuracy and skill-detector accuracy are distinct; no new detector
accuracy evaluation is reported here.

These results describe similar aggregate hardware exposure with changes in its
allocation to predicted skills. They do not support a claim that unfamiliar objects
or switched boxes materially reduce MTTF. The comparison applies to retained
object-handling executions; it excludes the no-pick attempts listed above.

## Files

- [summary.csv](summary.csv): all eight conditions, run counts, durations, failure
  probabilities, MTTF, leading sensitivity-ranked components, and changes relative
  to each policy's standard condition.
- [skill_contributions.csv](skill_contributions.csv): skill-specific failure
  contributions, percentage shares, durations, and occurrences per run.
- [component_exposures.csv](component_exposures.csv): component exposure and hazard
  accumulated using survival-weighted expected DTMC visits.
- [episode_metrics.csv](episode_metrics.csv): per-recording diagnostics. These
  individual-recording model results are not an arithmetic decomposition of the
  pooled DTMC result in the summary table.
- [solver_verification.json](solver_verification.json): exact PRISM and Storm
  verification status and versions for all five variation models.

The [main publication table](../paper_results.csv) and the 31-experiment publication
manifest are unchanged; these five variations are supplementary results.
