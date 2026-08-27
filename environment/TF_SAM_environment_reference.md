# Local TF_SAM Environment Reference

This file records the relevant packages installed in the local `TF_SAM` Conda environment used during development. It is a reference snapshot, not a strict environment lock file. For a new installation, follow the current SAM official setup instructions first.

## Platform

- Python: `3.12.4`
- Platform: `Linux x86_64 (glibc 2.35)`

## Relevant Package Versions

```text
matplotlib==3.10.3
numpy==1.26.4
opencv-python==4.11.0.86
pillow==11.2.1
torch==2.2.0+cu118
torchvision==0.17.0+cu118
```

`segment-anything` was not installed as a pip package in this environment; it was imported from local SAM source code. This repository follows the same approach by bundling the required SAM source under `third_party/segment-anything/`.
