"""Stage 1: generate camouflaged-object bounding boxes with Qwen2.5-VL."""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
qwen_utils_src = project_root / "third_party" / "qwen-vl-utils" / "src"
if qwen_utils_src.is_dir():
    sys.path.insert(0, str(qwen_utils_src))

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from local_paths import QWEN_MODEL_PATH
except ImportError:
    QWEN_MODEL_PATH = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
default_model_path = QWEN_MODEL_PATH or "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=project_root / "Imgs_or", type=Path)
    parser.add_argument("--output-dir", default=project_root / "Json_case1", type=Path)
    parser.add_argument("--model-path", default=default_model_path)
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Maximum images to process; 0 processes all images.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate JSON files that already exist.")
    return parser.parse_args()


def load_prompt(filename):
    return (project_root / "Qwen-Prompt" / filename).read_text(encoding="utf-8")


def parse_generated_json(text):
    """Parse plain JSON and JSON enclosed in a Markdown code fence."""
    content = text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else ""
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start_positions = [index for index in (content.find("["), content.find("{")) if index >= 0]
        if not start_positions:
            raise
        return json.JSONDecoder().raw_decode(content[min(start_positions) :])[0]


def main():
    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt = load_prompt("Sys_prompt.txt")
    user_prompt = load_prompt("User_prompt.txt")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device_map
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    processed = 0
    for image_path in sorted(args.image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        output_path = args.output_dir / f"{image_path.stem}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path.name} already exists")
            continue
        if args.limit and processed >= args.limit:
            break

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            prediction = parse_generated_json(output_text)
        except json.JSONDecodeError as error:
            print(f"[skip] {image_path.name}: invalid model JSON ({error})")
            continue

        output_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1
        print(f"[done] {image_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
