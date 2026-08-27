# UCOD-MKD: Qwen2.5-VL + SAM Camouflaged Object Segmentation

This repository implements a four-stage camouflaged object segmentation and student-training pipeline:

1. **Stage 1** uses Qwen2.5-VL to produce JSON bounding-box predictions from input images and prompts.
2. **Stage 2** uses these bounding boxes as prompts for Segment Anything Model (SAM) and produces binary segmentation masks.
3. **Stage 3** quality-tiers the SAM masks using candidate similarity, edge truncation, and fragmentation.
4. **Stage 4** prepares BeKD-style pseudo-label, pixel-confidence, and box-exterior background supervision, then trains a COD student network.

[[Pre_Map](https://drive.google.com/drive/folders/15HdilR7hWewk1rEqJOy3T_4URojImw-a?usp=drive_link)]
## Repository Layout

```text
UCOD_MKD_Github/
├── Stage_1.py                  # Stage 1 inference script
├── Stage_2.py                  # Stage 2 SAM segmentation script
├── Stage_3.py                  # Stage 3 mask quality-tiering script
├── Stage_4.py                  # Stage 4 supervision-package generator
├── Stage_4_train.py            # Stage 4 BeKD student-training entry point
├── Qwen-Prompt/
│   ├── Sys_prompt.txt          # System prompt
│   └── User_prompt.txt         # User prompt template
├── third_party/
│   ├── qwen-vl-utils/          # Bundled Qwen vision utilities source
│   └── segment-anything/       # Bundled SAM source
│   └── ucod_mkd_student/       # Student-network source (no weights/data)
├── checkpoints/                 # Place the downloaded SAM checkpoint here
├── Imgs_or/                    # Input images (create this directory)
├── Json_case1/                 # Stage 1 JSON output (created at runtime)
└── SAM_Masks/                  # Stage 2 single- and multi-mask outputs (created at runtime)
```

`User_prompt.txt` instructs the model to analyze the image scene, identify candidate animals, and return bounding boxes for camouflaged targets as JSON. `Sys_prompt.txt` instructs the model to return JSON without a Markdown code block.

## Environment Setup

Python 3.10 or later and a CUDA-enabled PyTorch environment are recommended.

### Stage 1: Qwen2.5-VL configuration

For a new Qwen environment, follow the current installation and compatibility guidance in the official [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL) first, especially for the PyTorch, CUDA, `transformers`, and `qwen-vl-utils` versions. Although this project runs **Qwen2.5-VL**, the official Qwen documentation should be treated as the source of truth for a compatible Qwen runtime.

This repository contains a development-environment reference at [environment/Qwen_environment_reference.md](environment/Qwen_environment_reference.md). It records the package versions from the local `Qwen` Conda environment, but it is not intended to replace the official setup instructions.

```bash
conda create -n ucod-mkd python=3.10 -y
conda activate ucod-mkd
# Install the PyTorch/CUDA combination recommended by the official Qwen documentation.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Use the PyTorch installation command compatible with your CUDA driver and the official Qwen guidance. CPU inference is possible, but it is generally too slow and memory-intensive for batch processing.

The required Qwen vision utility and SAM source code are included under `third_party/`; no separate `qwen-vl-utils` or `segment-anything` installation is required. The two stage scripts load these local copies automatically.

Model weights, input images, and generated outputs are intentionally excluded from version control. Place the Qwen model in a Hugging Face cache or update the Stage 1 model path, and download the SAM checkpoint before running Stage 2.

For local paths outside this repository, copy `local_paths.example.py` to `local_paths.py` and set `QWEN_MODEL_PATH` and/or `SAM_CHECKPOINT_PATH`. The local file is ignored by Git. Command-line arguments such as `--model-path` and `--checkpoint` take precedence over these defaults.

On the first run, the script downloads `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face. For offline use, download the model in advance and replace the model identifier in `Stage_1.py` with its local path.

## Prepare Input Data

Place images to be processed in `Imgs_or/`. The script iterates over all files in this directory and saves one JSON file per input image to `Json_case1/`, retaining the filename stem.

```bash
mkdir -p Imgs_or Json_case1
```

## Run Stage 1

`Stage_1.py` loads the prompt templates directly from `Qwen-Prompt/User_prompt.txt` and `Qwen-Prompt/Sys_prompt.txt`.

Run the following command from the repository root:

```bash
python Stage_1.py
```

By default, the script loads `Qwen/Qwen2.5-VL-3B-Instruct`, distributes the model automatically with `device_map="auto"`, and generates up to 512 new tokens for each image.

### One-image demo

Three small demonstration images have been placed in `Imgs_or/`. If the Qwen model has already been downloaded to a local directory, run one image with:

```bash
python Stage_1.py \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --limit 1 \
  --max-new-tokens 512
```

This writes the resulting bounding-box JSON to `Json_case1/`. Use `--overwrite` to regenerate an existing JSON file.

## Run Stage 2

### Stage 2: SAM configuration

For a new SAM environment, follow the current installation and checkpoint guidance in the official [Segment Anything repository](https://github.com/facebookresearch/segment-anything) first, especially for the PyTorch, CUDA, checkpoint, and model-type (`vit_h`, `vit_l`, or `vit_b`) combination. This repository bundles the required SAM source code, but you still need to configure a compatible PyTorch/CUDA environment and download a checkpoint.

This repository contains a development-environment reference at [environment/TF_SAM_environment_reference.md](environment/TF_SAM_environment_reference.md). It records package versions from the local `TF_SAM` Conda environment, but it is not intended to replace the official SAM setup instructions.

Download a SAM checkpoint from the official [Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints), then run:

```bash
mkdir -p checkpoints
# Put sam_vit_h_4b8939.pth in checkpoints/, then run:
python Stage_2.py
```

Alternatively, set `SAM_CHECKPOINT_PATH` in `local_paths.py` to use an existing checkpoint stored elsewhere.

Stage 2 reads images from `Imgs_or/` and the corresponding Stage 1 JSON files from `Json_case1/`. For every input image, it generates both single-mask and multi-mask SAM outputs under `SAM_Masks/`.

To run the matching one-image Stage 2 demo after Stage 1 completes, point the script at the same JSON directory and checkpoint:

```bash
python Stage_2.py \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth
```

Each Stage 1 JSON annotation should contain a `bbox_2d` field (or `bbox`) with coordinates in `[x_min, y_min, x_max, y_max]` order. For example:

```json
[
  {
    "bbox_2d": [190, 0, 1036, 672],
    "label": "camouflaged fish"
  }
]
```

Stage 2 always produces both output types:

```text
SAM_Masks/
├── single_mask/
│   └── <image_name>.png
└── multi_mask/
    ├── <image_name>_box1_candidate1_iou<score>.png
    ├── <image_name>_box1_candidate2_iou<score>.png
    └── <image_name>_box1_candidate3_iou<score>.png
```

`single_mask/` contains one merged binary mask per image: SAM produces one mask for each Qwen box, and Stage 2 unions these masks. `multi_mask/` contains all three SAM candidates for every Qwen box. The filename records the box index, candidate index, and SAM predicted-IoU score so later stages can select or combine candidates as needed.

## Run Stage 3

Stage 3 reads `SAM_Masks/multi_mask/` directly and writes one selected, merged mask per image to `Quality_Masks/High/`, `Quality_Masks/Normal/`, or `Quality_Masks/Low/`.

```bash
python Stage_3.py
```

For each bbox, Stage 3 first computes the mean pairwise IoU and SSIM of its three SAM candidates:

- Both means greater than `0.9`: **High** quality.
- Either mean below `0.6`: **Low** quality.
- Otherwise: inspect the selected candidate's edge-truncation count and connected-component count. Structurally valid masks are **Normal**; the remaining masks are downgraded to **Low**.

The selected candidate is the one with the highest SAM predicted-IoU in its filename. For images containing multiple boxes, their selected masks are merged. Any Low-quality box makes the final merged image Low; otherwise any Normal-quality box makes it Normal. `Quality_Masks/manifest.jsonl` records the metrics, selected candidate, and tier for every box.

## Output Format

Stage 1 produces one model-generated JSON file per input image. Stage 2 produces binary PNG masks; pixels belonging to predicted foreground objects have value 255 and background pixels have value 0.

## Run Stage 4

Stage 4 uses the Stage 3 tier in `Quality_Masks/manifest.jsonl`, the selected final mask, all three Stage 2 SAM candidates, and the Stage 1 bounding boxes:

- The three candidate masks are merged per candidate index for multi-box images.
- Their mean vote is converted to a candidate-consistency confidence map.
- `Pseudo_Labels/` uses the weak-label coding of `1` for foreground and `2` for background.
- `Reliable_Background/` labels only bbox-exterior pixels as background; bbox interiors are ignored.

Create the supervision package after Stage 3:

```bash
python Stage_4.py
```

The default output is:

```text
Stage_4_Data/
├── manifest.jsonl               # Image tier and paths used by training
├── train.txt
└── train/
    ├── Images/                  # Hard links to input images (copies if linking is unavailable)
    ├── Pseudo_Labels/           # Selected Stage 3 masks
    ├── Reliable_Background/     # Bbox-exterior reliable background
    ├── Confidence_Maps/         # Candidate-consistency confidence
    └── SAM_Candidates/          # Merged SAM candidates
```

`Stage_4_train.py` applies graded supervision to High, Normal, and Low pseudo-labels. Every branch also uses the reliable bbox-exterior background map. The bundled student source is adapted into a unified Stage 4 entry point.

### Stage 4: WSCOD environment

For Stage 4, first configure PyTorch and torchvision for the local CUDA driver, then install the remaining packages and resolve any final missing imports reported by the training entry point. This mirrors the practical setup of the original Scribble COD codebase rather than imposing an unnecessarily strict environment lock.

The local [`wscod` environment reference](environment/WSCOD_environment_reference.md) used Python 3.7.5, `torch==1.13.1+cu117`, `torchvision==0.14.1+cu117`, `timm==0.3.2`, and `tensorboardX==2.5.1`. A representative setup is:

```bash
conda create -n ucod-stage4 python=3.8 -y
conda activate ucod-stage4
# Choose the PyTorch/torchvision CUDA build compatible with this machine.
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python Stage_4_train.py --help
```

If the final command reports a missing module, install that named package in the same environment and rerun it. The original student-network project and its data conventions are documented in the [upstream Scribble COD repository](https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations).

Place the PVT-v2-B4 backbone checkpoint somewhere local, then run:

```bash
python Stage_4_train.py \
  --backbone-path /path/to/pvt_v2_b4.pth \
  --epochs 60 \
  --batch-size 8
```

The checkpoint is recommended but optional for a smoke test. Stage 4 requires `timm` and `tensorboardX`, both listed in `requirements.txt`. Training checkpoints are written to `Stage_4_Checkpoints/`.

## Troubleshooting

- `FileNotFoundError`: Run from the repository root, ensure that `Imgs_or/` and `Json_case1/` exist, and provide a valid SAM checkpoint path for Stage 2. The prompt files must remain in `Qwen-Prompt/`.
- Out of GPU memory: Stop other GPU workloads, use a smaller model, or reduce input image resolution.
- Model download failure: Check network access and Hugging Face permissions, or download the model beforehand and use a local model path.
