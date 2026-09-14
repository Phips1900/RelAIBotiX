# Publication experiment protocol

The publication manifest is `configs/experiments/paper.json`. It declares 31
experiments across real SO-ARM, LIBERO, simulated Franka, real Franka, and simulated
Stretch. SO-ARM MuJoCo and fault-injection experiments are outside the reported
scope.

## Fixed analysis rules

- Skill inference is mandatory for new recordings. The SO-ARM and LIBERO
  reproduction inputs retain the previously reviewed predictions; Franka and
  Stretch use the registered task-specific pretrained checkpoints.
- SO-ARM and LIBERO retain their complete recorded executions.
- Real and simulated Franka executions end after the final detected `Place` because
  reset motion is external to the demonstrated task. A complete episode is retained
  when the detector does not emit `Place`, as declared by `keep_missing_terminal`.
- The simulated bottle comparison uses only episodes marked successful so both mass
  conditions contain the intended payload exposure. Other experiments use the
  documented cleaned recording set; task success is not a primary reliability
  variable.
- The default and publication exposure model is `additive_normalized`.
- Sensitivity uses a tenfold perturbation of one base component probability at a
  time.
- Bottom-up and BDD fault-tree probabilities must agree numerically. Publication
  DTMC results are independently checked with exact PRISM and Storm execution.

## Portable data layout

The manifest paths are relative to a dataset root, which defaults to
`datasets/paper`. A downloaded or mounted dataset may be selected with
`--data-root`, the GUI’s **Dataset root** field, or `RELAIBOTIX_DATA_ROOT`.
Every input has a frozen SHA-256 checksum in the manifest.

```bash
relaibotix experiments run configs/experiments/paper.json \
  --data-root /path/to/relaibotix-paper-data \
  --output artifacts/paper-reproduction \
  --prism --prism-executable /path/to/prism \
  --storm --storm-executable docker://movesrwth/storm:stable
```

The compact frozen table is tracked in `results/paper`. Large HDF5 recordings,
checkpoints, intermediate behavioral tables, and generated model files are kept out
of Git and should be published as versioned dataset/release assets.

## Decomposition control

A controlled real-Franka sorting comparison used the same retained intervals,
component priors, reference values, and activity masks in two models: the full
four-skill model and a single `Operation` state with one fault tree. Under the
additive model, aggregate exposure is conserved. Failure probability and MTTF agree
within numerical rounding, policy ordering is unchanged, and all component ranks
are identical. This control therefore does not support a claim that decomposition
improves aggregate accuracy.

The supported benefit is additional resolution: the decomposed model identifies
the skill in which each component contributes most, retains observed loops and
retries, and allows the same skill fault trees to be reassessed under a changed
execution sequence. The private control analysis itself is intentionally not part
of the software release.
