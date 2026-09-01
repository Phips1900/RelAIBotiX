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
relaibotix behavior predicted.h5 --output artifacts/behavior
```

By default the analysis uses filtered predictions, then raw predictions, then
ground-truth skill labels. It reports per-segment and aggregate duration, velocity,
effort/torque, and joint traveled distance. Traveled distance is the sum of the
absolute position changes within a skill segment, rather than only the difference
between its first and last samples.

Outputs are written as CSV files plus `behavior.json`. Success detection and fault
injection are intentionally outside this release pipeline.

## Current case-study scope

The HDF5 and analysis interfaces are robot-independent. Existing pretrained
detectors cover mobile manipulation and Franka simulation. SO-ARM, real Franka,
LIBERO, and additional mobile checkpoints can be added without changing the
RelAIBotiX interface, provided their checkpoints and canonical feature schemas are
compatible with the detector package.

## License

MIT
