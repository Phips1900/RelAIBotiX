# RelAIBotiX

Dynamic, skill-resolved reliability assessment for AI-controlled robotic systems.

RelAIBotIX provides one portable workflow to:

1. validate or convert an HDF5 recording;
2. run a pretrained skill detector;
3. calculate behavioral metrics;
4. build reliability models with the DTMC, fault-tree, PRISM, BDD, and STORM backends.

All stages are available through the command-line interface, either independently
or as one reproducible run.

The frozen publication specification and compact result tables are available in
[configs/experiments/paper.json](configs/experiments/paper.json) and
[results/paper](results/paper). The mathematical model, assumptions, and source
traceability are documented in [docs/METHODS.md](docs/METHODS.md); the case-study
selection and reproduction protocol are documented in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Phips1900/RelAIBotiX.git
cd RelAIBotiX
python -m pip install -e .
```

For a non-editable installation from a tagged release, use `python -m pip install .`.

Install the separately maintained pretrained-detector runtime when skill inference
is needed:

```bash
python -m pip install -e '.[skill-detection]'
```

## Native desktop GUI

Install the optional PySide6 interface together with the skill-detector runtime:

```bash
python -m pip install -e '.[gui,skill-detection]'
relaibotix gui
```

The desktop application provides native file selection, robot-configuration and
checkpoint selection, HDF5 validation, background execution of the complete
pipeline, optional exact PRISM/STORM verification, and a concise results overview.
It invokes the same in-process workflow as the command-line interface; reliability
calculations are not reimplemented in the GUI.

After a run, the window switches to a result dashboard containing cumulative
traveled distance by component and physical unit, total duration by skill,
per-skill failure probability, and the ranked component-sensitivity result. Plot
toolbars provide zooming and export to publication-friendly image formats. The
setup panel collapses automatically so the figures use the available window space.
Additional behavioral views show stacked low/medium/high velocity and effort
exposure and an episode-selectable skill timeline. Sortable tables expose the skill,
component, reliability, and sensitivity values behind every visualization.

The **Experiment batch** workspace runs a complete publication manifest, can exclude
entries marked optional, and optionally verifies every generated model with exact
PRISM and STORM. It presents a combined sortable paper table plus policy-comparison
plots for failure probability and repeated-operation MTTF. The generated CSV,
Markdown, LaTeX, and provenance files remain the authoritative export artifacts.

The publication manifest contains portable relative dataset paths and SHA-256
checksums rather than machine-specific SSD paths. Select the released dataset root
in the GUI before running the batch.

Old paper recordings that already contain reviewed detector predictions can enable
**Use stored predictions (legacy paper reproduction only)**. This option is explicit
and disabled by default. New recordings must use a compatible registered detector
or a supplied checkpoint.

## HDF5 input

The canonical layout stores each episode independently:

```text
/data/demo_XXXXXX/
├── features
├── timestamps/sim
├── episode/index  # optional; retained for source alignment when available
├── validity/valid # optional; false samples are excluded from behavioral exposure
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

Invalid or documented no-pick episodes can be removed into a new canonical input
before skill inference. Source files remain untouched and episode keys are preserved
for traceability:

```bash
relaibotix h5 select recording.h5 selected.h5 \
  --exclude-episode demo_000027 \
  --exclude-episode demo_000028 \
  --drop-feature unavailable_measurement
```

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
detector. The simulated Franka bottle and sorting detectors intentionally use the
same eight position features, so the task must also be supplied or the detector
must be selected explicitly:

```bash
relaibotix skills infer canonical.h5 \
  --case-study franka_sim \
  --task bottle_task \
  --checkpoint-root /path/to/relaibotix-skill-detector/outputs \
  --output predicted.h5
```

The task-specific checkpoints retain their calibrated post-processing defaults:
10 minimum frames for the bottle task and 5 for sorting. Both also use the
calibrated Franka manipulation transition constraints. A different model can be
selected explicitly:

The same task checkpoints support real Franka recordings through zero-shot
transfer. Select `franka_real`; the detector then recognizes canonical files
whose root `source_format` identifies real Franka data and maps the recorded
gripper range from `[0, 1]` to the checkpoint convention `[-1, 1]` in memory:

```bash
relaibotix skills infer real_franka.h5 \
  --case-study franka_real \
  --task bottle_task \
  --checkpoint-root /path/to/relaibotix-skill-detector/outputs \
  --output predicted.h5
```

Bottle and sorting each reuse one checkpoint across simulation, real hardware,
policies, and payload variations. Model weights are not copied into this
repository. For an explicit custom checkpoint, `--input-profile auto` is the
default; `--input-profile real-franka` can force the adapter when older files do
not contain the expected source-format metadata.

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

By default, a measured motion component uses the additive normalized exposure model:

```text
effective exposure = active time
                   + normalized effort-time (when effort is available)
                   + normalized motion-time (when position is available)
hazard             = base failure rate × effective exposure
failure probability = 1 - exp(-hazard)
```

Each available term is used independently. For example, SO-ARM recordings have no
effort channel and therefore use active time plus normalized motion; they do not
silently lose the motion contribution. Components without either required reference
retain measured active time. Always-active electronics use elapsed skill time.

Several exposure calculations are available. `bands` assigns each active interval to
the configured low, medium, or high multiplier. `continuous` linearly interpolates
the factor at every active timestamp between the same configured thresholds and
multipliers, then caps it at the high multiplier. Distance is treated in the same
way using the mean distance per skill occurrence. The chosen method is saved in
`reliability.json`, each component row, and publication provenance.

Create the behavioral tables first, then build the per-skill fault trees and the
empirical DTMC:

```bash
relaibotix reliability artifacts/behavior/behavior.json \
  --config configs/robots/so_arm.json \
  --output artifacts/reliability \
  --exposure-model continuous \
  --sensitivity
```

Omit `--exposure-model` to use `additive_normalized`, the supported default. Select
`bands`, `continuous`, `normalized_product`, or `torque_distance` explicitly to
reproduce or compare the retained alternative calculations. Publication manifests
can override the model for individual experiments.

The optional direct normalized-product model is selected with:

```bash
relaibotix reliability artifacts/behavior/behavior.json \
  --config configs/robots/franka.json \
  --output artifacts/reliability-normalized \
  --exposure-model normalized_product
```

It evaluates fixed one-second physical windows independently of skill boundaries.
For each measured component it calculates RMS velocity, RMS effort, and absolute
traveled distance, divides each available quantity by 30% of its configured
official limit/reference, and directly multiplies the normalized values. Missing
telemetry dimensions are neutral: Franka uses velocity × effort × distance, while
LIBERO and SO-ARM use velocity × distance because their current recordings do not
contain effort. The resulting physical exposure is attributed to detector skills
after calculation. Components without suitable normalized references retain their
existing active-time exposure, and always-active electronics remain time-based.

The optional `torque_distance` model accumulates traveled joint distance weighted
by the cube of normalized effort. Velocity remains represented through
`abs(delta_position) = abs(velocity) * delta_time`, so motion is not counted twice:

```bash
relaibotix reliability behavior.json configs/robots/franka.json \
  --exposure-model torque_distance
```

The `additive_normalized` model preserves measured active operating time and adds
linear normalized effort-time and motion-time contributions:

```text
effective exposure = active time
                   + sum(abs(effort) / effort reference × delta time)
                   + sum(abs(delta position) / velocity reference)
```

References are the configured fraction of each component limit. The terms are not
centered or fitted, and the result must therefore be interpreted as a stress-adjusted
baseline model rather than a reproduction of lifetime at the reference condition.
Unavailable physical terms are omitted individually. Thus, position-only or
position-and-velocity recordings still contribute normalized motion, while an
available effort channel adds normalized load exposure.

### Nominal actuator failure probabilities

Robot-specific field-failure tables are not publicly available for SO-ARM 101 or
Stretch 3. Their actuator probabilities therefore use a transparent component-life
allocation rather than undocumented legacy constants. Probabilities are stored per
minute and converted from a nominal lifetime (L) as
`p = 1 - exp(-1 / (60 L))`.

- SO-ARM uses six identical STS3215 actuators (five joints and the gripper). The
  STS3215 specification documents a load-life qualification exceeding 100,000
  cycles at one-fifth stall torque, but this is not treated as an MTTF. In the
  absence of an STS3215 lifetime, each actuator uses the 50,000-hour generic servo
  reference published by Oriental Motor: `3.33333277777784e-7` per minute.
- Stretch uses four closed-loop stepper assemblies for its two wheels, lift, and
  telescoping arm. Because the platform uses low-ratio belt drives rather than
  harmonic gearheads, each uses the published generic 50,000-hour stepper bearing-life
  reference: `3.33333277777784e-7` per minute. Its Dynamixel-driven gripper uses the
  same generic 50,000-hour servo reference. The wrist and head are represented as
  assemblies of three and two series servos, giving `9.999995000001667e-7` and
  `6.666664444444938e-7` per minute, respectively.

These are cross-family nominal references, not measured SO-ARM or Stretch field
rates. The existing controller, power-supply, and camera rates remain separate
assumptions. Sources: [STS3215 specification](https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf),
[Oriental Motor service-life reference](https://www.orientalmotor.com/support/service-life.html),
and the [Stretch 3 hardware inventory](https://docs-arch.hello-robot.com/0.3/hardware/hardware_guide_stretch_3/).

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
the absolute change in overall failure probability. A publication-ready
`sensitivity_spider.svg` shows the system failure-probability ratio produced by each
one-at-a-time perturbation. Pass another factor explicitly,
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

On systems using the official STORM Docker image, no wrapper script is needed:

```bash
relaibotix reliability artifacts/behavior/behavior.json \
  --config configs/robots/so_arm.json \
  --output artifacts/reliability \
  --storm --storm-executable docker://movesrwth/storm:stable
```

RelAIBotiX also exports `model_repeated_runs.pm`. In that model, completing one
recorded run returns to the start state while modeled component failures remain
absorbing. Its expected accumulated time until failure is reported as
`repeated_run_mttf` in `reliability.json`. This represents repeated operation with
the measured mixture of skill sequences; it is separate from the failure
probability of one run.

The frozen publication experiment set is declared in one small manifest and can
be regenerated with:

```bash
relaibotix experiments run configs/experiments/paper.json \
  --data-root /path/to/relaibotix-paper-data \
  --output artifacts/paper-reproduction \
  --prism --prism-executable /path/to/prism \
  --storm --storm-executable docker://movesrwth/storm:stable
```

This writes per-experiment behavioral and reliability data plus combined CSV,
Markdown, and LaTeX tables and a provenance file containing input/configuration
hashes and solver versions. The manifest explicitly uses predictions already stored
in the legacy HDF5 recordings so the previous experiments can be recalculated. New
case-study data must use `relaibotix run`, which performs skill inference before the
behavioral and reliability stages.

The data root defaults to `datasets/paper` and can be overridden with the command
above, the GUI’s **Dataset root** field, or `RELAIBOTIX_DATA_ROOT`. Each input
checksum is verified before its experiment starts.

Each experiment can also declare `episode_selection` (`all` or `successful`),
`terminal_skill` (a skill name or ID), and `exclude_missing_terminal` (boolean).
These fields make the paper's run-selection and stopping rules explicit and are
copied into `provenance.json`. Their defaults retain every complete recording and
do not silently exclude episodes.

Set `keep_missing_terminal` to `true` when episodes containing the terminal skill
should be trimmed at that skill while retained pick-only episodes should keep their
complete observed duration.

## Current case-study scope

The HDF5 and analysis interfaces are robot-independent. Existing pretrained
detectors cover mobile manipulation and task-specific Franka simulation and
real-hardware inference. The publication reproduction retains the reviewed SO-ARM
and LIBERO predictions used in the preceding study; new recordings require a
compatible detector checkpoint.

The Hello Robot Stretch 3 configuration is available at
`configs/robots/hello_stretch.json`. It maps the logged wheel, lift, telescoping-arm,
wrist, gripper, and head mechanisms plus the always-active controller, power supply,
and camera. Multi-axis wrist and head measurements are combined into one reliability
component each: traveled distance is summed across axes, while velocity and effort
use the most heavily loaded axis at each timestep so elapsed time is counted once.
Its actuator failure probabilities use the documented derivation above and remain
explicit modeling assumptions pending platform-specific field data. The mobile
detector must run before behavioral or reliability analysis when an input contains
only unknown skill IDs.

## License

MIT. When using RelAIBotIX in academic work, cite the versioned software release
described in [CITATION.cff](CITATION.cff).
