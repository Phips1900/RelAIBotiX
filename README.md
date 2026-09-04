# RelAIBotiX

Dynamic reliability assessment for AI-controlled robotic systems.

This release branch is being reorganized around one portable workflow:

1. validate or convert an HDF5 recording;
2. run a pretrained skill detector;
3. calculate behavioral metrics;
4. build reliability models with the DTMC, fault-tree, PRISM, BDD, and STORM backends.

All stages are available through the command-line interface, either independently
or as one reproducible run.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Phips1900/RelAIBotiX.git
cd RelAIBotiX
python -m pip install -e .
```

Install the separately maintained pretrained-detector runtime when skill inference
is needed:

```bash
python -m pip install -e '.[skill-detection]'
```

## HDF5 input

The canonical layout stores each episode independently:

```text
/data/demo_XXXXXX/
├── features
├── timestamps/sim
├── episode/index
└── labels/skill_id
```

Feature names are stored on each `features` dataset. Inputs can be checked without
modification:

```bash
relaibotix h5 inspect recording.h5
relaibotix h5 validate recording.h5
relaibotix h5 validate recording.h5 --config configs/robots/so_arm.json
```

Supplying `--config` additionally checks every configured measurement against the
HDF5 feature schema. Missing measurements are errors; unused HDF5 channels are
reported as warnings so action commands and case-study metadata remain visible
without being treated as reliability inputs.

A supported flat legacy file can be converted to a separate canonical copy:

```bash
relaibotix h5 convert legacy.h5 canonical.h5
```

Neither validation nor conversion changes the source file.

## Complete pipeline

Run validation, optional flat-to-canonical conversion, mandatory skill inference,
behavioral analysis, fault trees, BDD, DTMC, PRISM export, and sensitivity analysis
with one command:

```bash
relaibotix run recording.h5 \
  --config configs/robots/hello_stretch.json \
  --checkpoint-root /path/to/detector/outputs \
  --output artifacts/mobile_act \
  --sensitivity
```

Add `--storm` to verify the generated PRISM model with STORM. The output directory
contains `predicted.h5`, the behavioral tables, reliability tables, sensitivity
results, and the PRISM model/property files. Existing prediction outputs are never
overwritten.

## Skill inference

Training lives in the separate
[relaibotix-skill-detector](https://github.com/Phips1900/relaibotix-skill-detector)
repository. RelAIBotiX only provides the inference connection:

```bash
relaibotix skills list --checkpoint-root /path/to/detector/outputs

relaibotix skills infer canonical.h5 \
  --checkpoint-root /path/to/detector/outputs \
  --output predicted.h5
```

The bundled registry lists all currently trained mobile and Franka-simulation
models. For HDF5-only inference, RelAIBotiX compares the recorded feature names
with the registered schemas and automatically selects the recommended time-series
detector. A different model can be selected explicitly:

```bash
relaibotix skills infer canonical.h5 \
  --detector mobile-transformer \
  --checkpoint-root /path/to/detector/outputs \
  --output predicted.h5
```

`RELAIBOTIX_CHECKPOINT_ROOT` can be used instead of repeating `--checkpoint-root`.
Model files remain in the detector release or future Hugging Face repository and
are not duplicated in this Git repository. `--checkpoint` remains available for
an explicit model path.

The checkpoint defines the architecture, ordered feature set, training
normalization, window alignment, and label taxonomy. These are deliberately not
duplicated as command-line arguments.

The detector copies the input and adds raw and filtered predictions below every
episode's `labels` group. The source HDF5 is never overwritten. Camera and hybrid
checkpoints can also be selected explicitly:

```bash
relaibotix skills infer canonical.h5 \
  --detector mobile-r3d18-d435i \
  --checkpoint-root /path/to/detector/outputs \
  --output predicted.h5 \
  --modality camera \
  --lerobot-root /path/to/aligned/videos
```

The video root is detector input for the camera and hybrid modalities; it is not a
RelAIBotiX release dataset or conversion requirement.

## Behavioral analysis

```bash
relaibotix behavior predicted.h5 \
  --config configs/robots/so_arm.json \
  --output artifacts/behavior
```

By default the analysis uses filtered predictions, then raw predictions, then
ground-truth skill labels. It reports per-segment and aggregate duration, velocity,
effort/torque, and joint traveled distance. Traveled distance is the sum of the
absolute position changes within a skill segment, rather than only the difference
between its first and last samples. A frame label owns the interval from that frame
to the next one (left-endpoint attribution), so time and motion at skill boundaries
are not discarded. Mobile recordings additionally produce separate
`base_metrics` and `base_summary` tables containing planar path length, wrapped yaw
travel, linear speed, and angular speed. These platform metrics are not treated as
an additional fault-tree component.

Outputs are written as CSV files plus `behavior.json`. Success detection and fault
injection are intentionally outside this release pipeline.

## Reliability foundation

Robot configurations in `configs/robots` use one versioned schema for robot identity,
typed components, HDF5 feature mappings, probabilities, redundancy, and exposure
assumptions. Redundancy is always explicit:

```bash
relaibotix config validate configs/robots/so_arm.json
```

```json
{
  "type": "controller",
  "always_active": true,
  "failure_probability": 1e-6,
  "redundancy": {"copies": 3, "mode": "parallel"}
}
```

The reliability package provides two fault-tree evaluators over the same validated
model:

- a traditional bottom-up evaluator for ordinary trees;
- an exact reduced ordered BDD evaluator, including trees where a basic event is
  referenced by more than one gate.

The velocity, effort, and distance adjustments are an expert-assumption model, not
fixed physical constants. Each robot configuration records the thresholds,
multipliers, and provenance used for a calculation:

```json
{
  "exposure_assumptions": {
    "source": "example_assumption_set_requires_expert_review",
    "velocity_active": 0.03,
    "effort_active": 0.1,
    "velocity_bands": [0.5, 1.0],
    "effort_bands": [0.2, 0.6],
    "velocity_multipliers": [1.0, 1.5, 2.0],
    "effort_multipliers": [1.0, 1.5, 2.0],
    "distance_multipliers": [1.0, 1.5, 2.0]
  }
}
```

These example values require review by a domain expert. The behavior and reliability
commands must use the same robot configuration; a recorded threshold mismatch is
rejected instead of silently mixing assumption sets.

For a motion component, RelAIBotiX applies the configured assumptions as:

```text
effective exposure = velocity-weighted time × effort factor × distance factor
hazard             = base failure rate × effective exposure
failure probability = 1 - exp(-hazard)
```

This is an explicit relative-exposure model. RelAIBotiX does not claim that the
example thresholds or multipliers are universally valid robot parameters.

Create the behavioral tables first, then build the per-skill fault trees and the
empirical DTMC:

```bash
relaibotix reliability artifacts/behavior/behavior.json \
  --config configs/robots/so_arm.json \
  --output artifacts/reliability \
  --sensitivity
```

This writes the component exposure and failure calculations, bottom-up and BDD
skill probabilities, the solved system DTMC, and `model.pm`/`model.pctl` for PRISM.
The DTMC's `done` probability is reported as *completion without modeled failure*;
it is not empirical task-success detection.
Every hazard calculation retains its base probability, time basis, per-skill-execution
active exposure, traveled distance, velocity weighting, effort factor, and final
probability for later comparisons. Motion components can classify their mean traveled
distance per skill occurrence into low, medium, and high bands:

```json
{
  "distance_thresholds": [0.5, 1.0],
  "distance_unit": "radian"
}
```

The lower threshold starts the medium band and the upper threshold starts the high
band. Components without configured thresholds retain a neutral distance factor of
`1.0`. Thresholds use the physical units of the corresponding position signal and
are never normalized against the analyzed dataset.

`--sensitivity` performs the component importance analysis used by the project. It
multiplies one component's base failure probability by ten, reruns the complete
fault-tree and DTMC calculation, restores that component, and repeats for every
component. The resulting `sensitivity.csv` and `sensitivity.json` rank components by
the absolute change in overall failure probability. Pass another factor explicitly,
for example `--sensitivity 5`, when required. Exposure measurements and all other
component probabilities remain unchanged during each perturbation.

If the `storm` executable is installed, the generated PRISM model can be verified
with the optional STORM backend:

```bash
relaibotix reliability artifacts/behavior/behavior.json \
  --config configs/robots/so_arm.json \
  --output artifacts/reliability \
  --storm
```

Exact arithmetic is the default for both PRISM and STORM. Use
`--approximate-solvers` only for exploratory runs, or `--storm-executable` and
`--prism-executable` when the binaries are not on the normal executable path.
Both tools consume the exported `.pm` and `.pctl` models and are checked against
the internal solver.

RelAIBotiX also exports `model_repeated_runs.pm`. In that model, completing one
recorded run returns to the start state while modeled component failures remain
absorbing. Its expected accumulated time until failure is reported as
`repeated_run_mttf` in `reliability.json`. This represents repeated operation with
the measured mixture of skill sequences; it is separate from the failure
probability of one run.

The complete legacy paper experiment set is declared in one small manifest and can
be regenerated with:

```bash
relaibotix experiments run configs/experiments/paper.json \
  --output artifacts/paper_validation \
  --prism --prism-executable /path/to/prism
```

This writes per-experiment behavioral and reliability data plus combined CSV,
Markdown, and LaTeX tables and a provenance file containing input/configuration
hashes and solver versions. The manifest explicitly uses predictions already stored
in the legacy HDF5 recordings so the previous experiments can be recalculated. New
case-study data must use `relaibotix run`, which performs skill inference before the
behavioral and reliability stages.

## Current case-study scope

The HDF5 and analysis interfaces are robot-independent. Existing pretrained
detectors cover mobile manipulation and Franka simulation. SO-ARM, real Franka,
and LIBERO remain intentionally absent from the registry until their detector
checkpoints exist. They can then be added without changing the RelAIBotiX interface,
provided their checkpoints and canonical feature schemas are compatible with the
detector package.

The Hello Robot Stretch 3 configuration is available at
`configs/robots/hello_stretch.json`. It maps the logged wheel, lift, telescoping-arm,
wrist, gripper, and head mechanisms plus the always-active controller, power supply,
and camera. Multi-axis wrist and head measurements are combined into one reliability
component each: traveled distance is summed across axes, while velocity and effort
use the most heavily loaded axis at each timestep so elapsed time is counted once.
Its current failure probabilities and exposure bands are
explicitly provisional and require expert review. The available mobile ACT rollout
file is structurally valid but contains only unknown skill IDs, so detector inference
must run before behavioral or reliability analysis.

## License

MIT
