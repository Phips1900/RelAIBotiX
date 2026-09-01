# Skill inference

RelAIBotiX contains inference support only. Model development and training are
maintained in the separate skill-detector repository.

Inference reads a canonical HDF5 file and writes one label per sample to
`/skills/predicted`:

```bash
relaibotix skills infer canonical.h5 \
  --checkpoint artifacts/checkpoints/skill_detector.ckpt \
  --features x y z ox oy oz ow gripper_state
```

Feature names must be supplied in the exact order used during training. Window
size and class count are read from the checkpoint when available. Inference is
performed independently for every episode; short episodes are edge-padded by
default and no window crosses an episode boundary.
