"""Stage 4: prepare BeKD-style student-training supervision from Stages 1--3."""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CANDIDATE_PATTERN = re.compile(
    r"^(?P<image>.+)_box(?P<box>\d+)_candidate(?P<candidate>\d+)_iou[0-9.]+\.png$"
)
QUALITY_VALUE = {"Low": 0, "Normal": 1, "High": 2}
LEGACY_TIER_NAMES = {"Medium": "Normal"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "Imgs_or")
    parser.add_argument("--json-dir", type=Path, default=ROOT / "Json_case1")
    parser.add_argument("--multi-mask-dir", type=Path, default=ROOT / "SAM_Masks" / "multi_mask")
    parser.add_argument("--quality-dir", type=Path, default=ROOT / "Quality_Masks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Stage_4_Data")
    parser.add_argument("--image-mode", choices=("link", "copy", "none"), default="link",
                        help="How to place source images under Stage4_Data/train/Imgs.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0,
                        help="Gaussian smoothing for Refer/guass_weight.py-style pixel weights; 0 disables it.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def binary_mask(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read mask: {path}")
    return image >= 128


def candidate_groups(mask_dir):
    groups = defaultdict(lambda: defaultdict(dict))
    for path in mask_dir.glob("*.png"):
        match = CANDIDATE_PATTERN.match(path.name)
        if match:
            data = match.groupdict()
            groups[data["image"]][int(data["box"])][int(data["candidate"])] = path
    return groups


def find_image(image_dir, stem):
    matches = sorted(path for path in image_dir.glob(f"{stem}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return matches[0] if matches else None


def load_boxes(json_path, shape):
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("objects", data) if isinstance(data, dict) else []
    if isinstance(records, dict):
        records = list(records.values())
    boxes = []
    height, width = shape
    for record in records:
        if not isinstance(record, dict):
            continue
        box = record.get("bbox_2d", record.get("bbox"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(float(value)) for value in box)
        x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def copy_or_link(source, target, mode):
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "link":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    if mode != "none":
        shutil.copy2(source, target)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main():
    args = parse_args()
    for required in (args.image_dir, args.multi_mask_dir, args.quality_dir):
        if not required.is_dir():
            raise FileNotFoundError(required)

    output = args.output_dir
    directories = {
        name: output / "train" / name
        for name in ("Images", "Pseudo_Labels", "Reliable_Background", "Confidence_Maps")
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    quality_manifest = {}
    manifest_path = args.quality_dir / "manifest.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        quality_manifest[record["image"]] = record

    groups = candidate_groups(args.multi_mask_dir)
    records = []
    for stem, boxes_by_index in sorted(groups.items()):
        quality = quality_manifest.get(stem)
        if not quality:
            print(f"[skip] {stem}: absent from Stage 3 manifest")
            continue
        source_tier = quality["tier"]
        tier = LEGACY_TIER_NAMES.get(source_tier, source_tier)
        image_path = find_image(args.image_dir, stem)
        final_path = args.quality_dir / source_tier / f"{stem}.png"
        if image_path is None or not final_path.exists():
            print(f"[skip] {stem}: source image or final mask is missing")
            continue
        final_mask = binary_mask(final_path)
        candidate_masks = []
        try:
            for candidate in (1, 2, 3):
                per_box = [binary_mask(candidates[candidate]) for _, candidates in sorted(boxes_by_index.items())]
                candidate_masks.append(np.logical_or.reduce(per_box))
        except (KeyError, ValueError) as error:
            print(f"[skip] {stem}: incomplete candidates ({error})")
            continue
        if any(mask.shape != final_mask.shape for mask in candidate_masks):
            print(f"[skip] {stem}: candidate and final-mask dimensions differ")
            continue

        # Refer/fuse_mask.py: p is the mean vote of the three SAM candidates;
        # Refer uses 1 - binary entropy as the confidence map.
        probability = np.mean(np.stack(candidate_masks, axis=0), axis=0)
        clipped = np.clip(probability, 1e-10, 1 - 1e-10)
        pixel_weight = 1.0 + clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
        if args.gaussian_sigma > 0:
            pixel_weight = cv2.GaussianBlur(pixel_weight.astype(np.float32), (0, 0), args.gaussian_sigma)
        pixel_weight = np.clip(pixel_weight, 0, 1)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        boxes = load_boxes(args.json_dir / f"{stem}.json", final_mask.shape)
        # Code-compatible labels: 1 foreground, 2 background, 0 unknown/ignore.
        pseudo = np.where(final_mask, 1, 2).astype(np.uint8)
        box_background = np.full(final_mask.shape, 2, dtype=np.uint8)
        for x0, y0, x1, y1 in boxes:
            box_background[y0:y1, x0:x1] = 0

        candidate_directory = output / "train" / "SAM_Candidates"
        candidate_targets = [candidate_directory / f"Candidate_{index}" / f"{stem}.png" for index in (1, 2, 3)]
        for path in (candidate_directory / "Candidate_1", candidate_directory / "Candidate_2", candidate_directory / "Candidate_3"):
            path.mkdir(parents=True, exist_ok=True)
        targets = [
            directories["Pseudo_Labels"] / f"{stem}.png",
            directories["Reliable_Background"] / f"{stem}.png",
            directories["Confidence_Maps"] / f"{stem}.png",
            *candidate_targets,
        ]
        if not args.overwrite and any(path.exists() for path in targets):
            print(f"[skip] {stem}: output exists")
            continue
        copy_or_link(image_path, directories["Images"] / f"{stem}{image_path.suffix.lower()}", args.image_mode)
        write_image(targets[0], pseudo)
        write_image(targets[1], box_background)
        write_image(targets[2], np.rint(pixel_weight * 255).astype(np.uint8))
        for target, mask in zip(candidate_targets, candidate_masks):
            write_image(target, mask.astype(np.uint8) * 255)
        records.append({
            "image": stem,
            "source_image": image_path.name,
            "tier": tier,
            "tier_value": QUALITY_VALUE[tier],
            "bbox_count": len(boxes),
            "pseudo_label": f"Pseudo_Labels/{stem}.png",
            "reliable_background": f"Reliable_Background/{stem}.png",
            "confidence_map": f"Confidence_Maps/{stem}.png",
        })
        print(f"[done] {stem}: {tier}, {len(boxes)} box(es)")

    (output / "train.txt").write_text("\n".join(record["image"] for record in records) + ("\n" if records else ""), encoding="utf-8")
    (output / "manifest.jsonl").write_text("\n".join(json.dumps(record) for record in records) + ("\n" if records else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
