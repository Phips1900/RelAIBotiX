# Skill inference adapter

Skill-detector training and model definitions are maintained in the separate
`relaibotix-skill-detector` repository. This package is a thin inference adapter;
it does not duplicate detector architectures or training code.

```bash
relaibotix skills list --checkpoint-root /path/to/detector/outputs

relaibotix skills infer canonical.h5 \
  --checkpoint-root /path/to/detector/outputs \
  --output predicted.h5
```

Automatic selection compares the HDF5 feature names with the schemas in
`checkpoints.json` and chooses the recommended time-series model. Use `--detector`
to choose a different registered model or `--checkpoint` to bypass the registry.

The detector reads feature order, normalization, architecture, window alignment,
and label taxonomy from the checkpoint. It copies the source HDF5 and writes raw
and minimum-duration-filtered predictions to each episode's `labels` group.

Camera and hybrid models additionally require the aligned video dataset:

```bash
relaibotix skills infer canonical.h5 \
  --detector mobile-hybrid-lstm-r3d18-d435i \
  --checkpoint-root /path/to/detector/outputs \
  --output predicted.h5 \
  --modality hybrid \
  --lerobot-root /path/to/aligned/videos
```
