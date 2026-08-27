"""Stage 3: quality-tier the Stage 2 SAM masks using similarity and mask structure."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


project_root = Path(__file__).resolve().parent
CANDIDATE_PATTERN = re.compile(
    r"^(?P<image>.+)_box(?P<box>\d+)_candidate(?P<candidate>\d+)_iou(?P<score>[0-9.]+)\.png$"
)
QUALITY_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "SAM_Masks" / "multi_mask",
        help="Stage 2 multi-mask output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Quality_Masks",
        help="Directory for quality-tiered final masks and the manifest.",
    )
    parser.add_argument("--high-threshold", type=float, default=0.9)
    parser.add_argument("--low-threshold", type=float, default=0.6)
    parser.add_argument("--edge-margin", type=int, default=10)
    parser.add_argument("--edge-pixels", type=int, default=20)
    parser.add_argument("--max-edge-touch", type=int, default=1)
    parser.add_argument("--max-components", type=int, default=49)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_binary_mask(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read mask: {path}")
    return image >= 128


def mask_iou(mask_a, mask_b):
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0


def mask_ssim(mask_a, mask_b):
    minimum_size = min(mask_a.shape)
    if minimum_size < 7:
        return float(np.array_equal(mask_a, mask_b))
    window_size = min(7, minimum_size if minimum_size % 2 else minimum_size - 1)
    return float(structural_similarity(mask_a.astype(np.uint8), mask_b.astype(np.uint8), data_range=1, win_size=window_size))


def average_similarity(masks):
    pairs = ((0, 1), (0, 2), (1, 2))
    ious = [mask_iou(masks[first], masks[second]) for first, second in pairs]
    ssims = [mask_ssim(masks[first], masks[second]) for first, second in pairs]
    return float(np.mean(ious)), float(np.mean(ssims))


def edge_truncation_count(mask, margin, min_foreground_pixels):
    height, width = mask.shape
    margin = min(margin, max(0, height - 1), max(0, width - 1))
    rows = (mask[margin, :], mask[height - 1 - margin, :])
    columns = (mask[:, margin], mask[:, width - 1 - margin])
    return int(sum(strip.sum() > min_foreground_pixels for strip in (*rows, *columns)))


def fragmentation_count(mask):
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(num_labels - 1)  # Label 0 is the background.


def group_candidates(input_dir):
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(input_dir.glob("*.png")):
        match = CANDIDATE_PATTERN.match(path.name)
        if not match:
            continue
        info = match.groupdict()
        groups[info["image"]][int(info["box"])].append(
            {"path": path, "candidate": int(info["candidate"]), "sam_iou": float(info["score"])}
        )
    return groups


def tier_box(candidates, args):
    candidates = sorted(candidates, key=lambda item: item["candidate"])
    if len(candidates) != 3 or [item["candidate"] for item in candidates] != [1, 2, 3]:
        raise ValueError("expected exactly candidates 1, 2, and 3")
    masks = [load_binary_mask(item["path"]) for item in candidates]
    if len({mask.shape for mask in masks}) != 1:
        raise ValueError("candidate mask sizes do not match")

    average_iou, average_ssim = average_similarity(masks)
    selected_index = max(range(3), key=lambda index: candidates[index]["sam_iou"])
    selected_mask = masks[selected_index]

    if average_iou > args.high_threshold and average_ssim > args.high_threshold:
        tier = "High"
        edge_count = None
        component_count = None
    elif average_iou < args.low_threshold or average_ssim < args.low_threshold:
        tier = "Low"
        edge_count = None
        component_count = None
    else:
        edge_count = edge_truncation_count(selected_mask, args.edge_margin, args.edge_pixels)
        component_count = fragmentation_count(selected_mask)
        if edge_count <= args.max_edge_touch and component_count <= args.max_components:
            tier = "Medium"
        else:
            tier = "Low"

    return selected_mask, {
        "tier": tier,
        "average_iou": average_iou,
        "average_ssim": average_ssim,
        "selected_candidate": candidates[selected_index]["candidate"],
        "selected_sam_iou": candidates[selected_index]["sam_iou"],
        "edge_truncation_count": edge_count,
        "fragmentation_count": component_count,
    }


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Stage 2 multi-mask directory does not exist: {args.input_dir}")

    for tier in QUALITY_ORDER:
        (args.output_dir / tier).mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    groups = group_candidates(args.input_dir)
    manifest = []

    for image_name, boxes in groups.items():
        output_paths = [args.output_dir / tier / f"{image_name}.png" for tier in QUALITY_ORDER]
        if not args.overwrite and any(path.exists() for path in output_paths):
            print(f"[skip] {image_name}: output already exists")
            continue

        selected_masks = []
        box_records = []
        for box_index, candidates in sorted(boxes.items()):
            try:
                selected_mask, record = tier_box(candidates, args)
            except ValueError as error:
                print(f"[skip] {image_name}, box {box_index}: {error}")
                continue
            record["box_index"] = box_index
            selected_masks.append(selected_mask)
            box_records.append(record)

        if not selected_masks:
            continue
        if len({mask.shape for mask in selected_masks}) != 1:
            print(f"[skip] {image_name}: selected box mask sizes do not match")
            continue

        # A composite containing any Low box is Low; otherwise Medium takes precedence over High.
        image_tier = min((record["tier"] for record in box_records), key=QUALITY_ORDER.get)
        composite_mask = np.logical_or.reduce(selected_masks).astype(np.uint8) * 255
        cv2.imwrite(str(args.output_dir / image_tier / f"{image_name}.png"), composite_mask)
        manifest.append({"image": image_name, "tier": image_tier, "boxes": box_records})
        print(f"[done] {image_name}: {image_tier} ({len(box_records)} box(es))")

    manifest_path.write_text("\n".join(json.dumps(record) for record in manifest) + ("\n" if manifest else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
