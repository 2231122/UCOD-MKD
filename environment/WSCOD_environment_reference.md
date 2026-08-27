# Stage 4 WSCOD Environment Reference

This is a reference snapshot of the local `wscod` environment used to prepare and import-check Stage 4. It is not a lock file and does not replace the upstream installation instructions.

| Component | Reference version |
| --- | --- |
| Python | 3.7.5 |
| PyTorch | 1.13.1+cu117 |
| torchvision | 0.14.1+cu117 |
| CUDA runtime reported by PyTorch | 11.7 |
| timm | 0.3.2 |
| tensorboardX | 2.5.1 |
| einops | 0.4.1 |
| OpenCV | 4.7.0 |
| NumPy | 1.21.6 |
| Pillow | 9.5.0 |
| scikit-image | 0.19.2 |

## Recommended setup order

1. Create a dedicated Python environment. Python 3.7 is the historical reference; Python 3.8--3.10 is also a reasonable starting point if a compatible PyTorch/CUDA build is available.
2. Install PyTorch and torchvision first, selecting the CUDA build compatible with the machine and GPU driver. The reference pair is `torch==1.13.1+cu117` and `torchvision==0.14.1+cu117`.
3. Install this repository's `requirements.txt`.
4. Run `python Stage_4_train.py --help` to confirm that the Stage 4 student network imports. If a missing-package error remains, install the named package in the same environment and rerun the command.
5. Supply `--backbone-path /path/to/pvt_v2_b4.pth` before full training.

The Stage 4 student source derives from the Scribble COD codebase. Consult the upstream repository for its original dataset and training assumptions: [Weakly-Supervised Camouflaged Object Detection with Scribble Annotations](https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations).
