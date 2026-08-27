"""Stage 2: convert Qwen bounding-box predictions into SAM segmentation masks."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent
sam_source_dir = project_root / "third_party" / "segment-anything"
if sam_source_dir.is_dir():
    sys.path.insert(0, str(sam_source_dir))

from segment_anything import SamPredictor, sam_model_registry

try:
    from local_paths import SAM_CHECKPOINT_PATH
except ImportError:
    SAM_CHECKPOINT_PATH = None

default_checkpoint = Path(SAM_CHECKPOINT_PATH) if SAM_CHECKPOINT_PATH else project_root / "checkpoints" / "sam_vit_h_4b8939.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=project_root / "Imgs_or", type=Path, help="Directory containing input images.")
    parser.add_argument("--json-dir", default=project_root / "Json_case1", type=Path, help="Directory containing Stage 1 JSON files.")
    parser.add_argument("--output-dir", default=project_root / "SAM_Masks", type=Path, help="Directory for Stage 2 mask outputs.")
    parser.add_argument(
        "--checkpoint",
        default=default_checkpoint,
        type=Path,
        help="Path to a SAM checkpoint.",
    )
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def annotations_to_boxes(data):
    """Extract [x_min, y_min, x_max, y_max] boxes from common Stage 1 JSON layouts."""
    if isinstance(data, dict) and "objects" in data:
        data = data["objects"]

    if isinstance(data, dict):
        items = list(data.values()) if "bbox_2d" not in data and "bbox" not in data else [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    boxes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d", item.get("bbox"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            boxes.append([float(value) for value in box])
        except (TypeError, ValueError):
            continue
    return boxes


def clamp_boxes(boxes, image_height, image_width):
    valid_boxes = []
    for x_min, y_min, x_max, y_max in boxes:
        x_min = max(0, min(image_width, x_min))
        y_min = max(0, min(image_height, y_min))
        x_max = max(0, min(image_width, x_max))
        y_max = max(0, min(image_height, y_max))
        if x_max > x_min and y_max > y_min:
            valid_boxes.append([x_min, y_min, x_max, y_max])
    return valid_boxes


def main():
    args = parse_args()
    image_dir = args.image_dir
    json_dir = args.json_dir
    output_dir = args.output_dir
    single_mask_dir = output_dir / "single_mask"
    multi_mask_dir = output_dir / "multi_mask"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Stage 1 JSON directory does not exist: {json_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {args.checkpoint}")

    single_mask_dir.mkdir(parents=True, exist_ok=True)
    multi_mask_dir.mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    for json_path in sorted(json_dir.glob("*.json")):
        image_path = next(
            (path for path in image_dir.glob(f"{json_path.stem}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
            None,
        )
        if image_path is None:
            print(f"[skip] No image found for {json_path.name}")
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"[skip] Invalid JSON in {json_path.name}: {error}")
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[skip] Cannot read {image_path}")
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        boxes = clamp_boxes(annotations_to_boxes(data), height, width)
        if not boxes:
            print(f"[skip] No valid bounding boxes in {json_path.name}")
            continue

        predictor.set_image(image)
        input_boxes = torch.tensor(boxes, dtype=torch.float32, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        # Single-mask mode: SAM returns one mask per Qwen bounding box.
        single_masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        merged_single_mask = single_masks[:, 0].any(dim=0).to(torch.uint8).cpu().numpy() * 255
        Image.fromarray(merged_single_mask).save(single_mask_dir / f"{json_path.stem}.png")

        # Multi-mask mode: SAM returns all three candidates for every Qwen bounding box.
        multi_masks, multi_scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )
        for box_index, (box_masks, box_scores) in enumerate(zip(multi_masks, multi_scores), start=1):
            for candidate_index, (mask, score) in enumerate(zip(box_masks, box_scores), start=1):
                mask_image = mask.to(torch.uint8).cpu().numpy() * 255
                output_name = f"{json_path.stem}_box{box_index}_candidate{candidate_index}_iou{score.item():.3f}.png"
                Image.fromarray(mask_image).save(multi_mask_dir / output_name)

        print(f"[done] {json_path.name}: {len(boxes)} box(es) -> single mask + {len(boxes) * 3} multi-mask candidates")


if __name__ == "__main__":
    main()
