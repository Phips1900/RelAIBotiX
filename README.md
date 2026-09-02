# RelAIBotiX

Dynamic reliability assessment for AI-controlled robotic systems.

This release branch is being reorganized around one portable workflow:

1. validate or convert an HDF5 recording;
2. run a pretrained skill detector;
3. calculate behavioral metrics;
4. build reliability models with the DTMC, fault-tree, PRISM, BDD, and STORM backends.

The first three stages are available through the new command-line interface. The
reliability backends are currently being migrated from the legacy research code.

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
```

A supported flat legacy file can be converted to a separate canonical copy:

```bash
relaibotix h5 convert legacy.h5 canonical.h5
```

Neither validation nor conversion changes the source file.

## Skill inference

Training lives in the separate
[relaibotix-skill-detector](https://github.com/Phips1900/relaibotix-skill-detector)
repository. RelAIBotiX only provides the inference connection:

```bash
relaibotix skills infer canonical.h5 \
  --checkpoint /path/to/best.pt \
  --output predicted.h5
```

The checkpoint defines the architecture, ordered feature set, training
normalization, window alignment, and label taxonomy. These are deliberately not
duplicated as command-line arguments.

The detector copies the input and adds raw and filtered predictions below every
episode's `labels` group. The source HDF5 is never overwritten. Camera and hybrid
checkpoints can also be selected explicitly:

```bash
relaibotix skills infer canonical.h5 \
  --checkpoint /path/to/camera-or-hybrid.pt \
  --output predicted.h5 \
  --modality camera \
  --lerobot-root /path/to/aligned/videos
```

The video root is detector input for the camera and hybrid modalities; it is not a
RelAIBotiX release dataset or conversion requirement.

## Behavioral analysis

```bash
relaibotix behavior predicted.h5 \
  --config config_files/robots/so_arm_config.json \
  --output artifacts/behavior
```

By default the analysis uses filtered predictions, then raw predictions, then
ground-truth skill labels. It reports per-segment and aggregate duration, velocity,
effort/torque, and joint traveled distance. Traveled distance is the sum of the
absolute position changes within a skill segment, rather than only the difference
between its first and last samples.

Outputs are written as CSV files plus `behavior.json`. Success detection and fault
injection are intentionally outside this release pipeline.

## Reliability foundation

Robot configurations in `config_files/robots` define component failure
probabilities and redundancy. Existing Boolean redundancy values remain supported:
`true` means two identical copies whose combined loss is an AND event. New configs
can state the copy count explicitly:

```json
{
  "failure_probability": 1e-6,
  "redundancy": {"copies": 3}
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
  --config config_files/robots/so_arm_config.json \
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
  --config config_files/robots/so_arm_config.json \
  --output artifacts/reliability \
  --storm
```

Use `--storm-exact` for STORM's exact mode or `--storm-executable` when the binary
is not on the normal executable path. PRISM remains a first-class exported backend;
STORM consumes the same `.pm` and `.pctl` files.

## Current case-study scope

The HDF5 and analysis interfaces are robot-independent. Existing pretrained
detectors cover mobile manipulation and Franka simulation. SO-ARM, real Franka,
LIBERO, and additional mobile checkpoints can be added without changing the
RelAIBotiX interface, provided their checkpoints and canonical feature schemas are
compatible with the detector package.

## License

MIT
