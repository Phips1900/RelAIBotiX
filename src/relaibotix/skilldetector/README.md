# Skill inference adapter

Skill-detector training and model definitions are maintained in the separate
`relaibotix-skill-detector` repository. This package is a thin inference adapter;
it does not duplicate detector architectures or training code.

```bash
relaibotix skills infer canonical.h5 \
  --checkpoint /path/to/best.pt \
  --output predicted.h5
```

The detector reads feature order, normalization, architecture, window alignment,
and label taxonomy from the checkpoint. It copies the source HDF5 and writes raw
and minimum-duration-filtered predictions to each episode's `labels` group.

Camera and hybrid models additionally require the aligned video dataset:

```bash
relaibotix skills infer canonical.h5 \
  --checkpoint /path/to/best.pt \
  --output predicted.h5 \
  --modality hybrid \
  --lerobot-root /path/to/aligned/videos
```
