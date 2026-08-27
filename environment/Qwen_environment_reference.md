# Local Qwen Environment Reference

This file records the relevant packages installed in the local `Qwen` Conda environment used during development. It is a reference snapshot, not a strict environment lock file. For a new installation, follow the current Qwen official setup instructions first.

## Platform

- Python: `3.10.18`
- Platform: `Linux x86_64 (glibc 2.35)`

## Relevant Package Versions

```text
accelerate==1.11.0
av==16.0.1
huggingface-hub==0.36.0
numpy==1.26.2
packaging==25.0
pillow==11.3.0
qwen-vl-utils==0.0.14
requests==2.28.1
safetensors==0.6.2
tokenizers==0.21.4
torch==2.1.1+cu121
torchvision==0.16.1+cu121
transformers==4.49.0
```

`opencv-python` was not installed in this environment. Install it from this repository's `requirements.txt` before running Stage 2.
