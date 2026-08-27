"""Train the BeKD_WSCOD_2 student network on the Stage 4 supervision package."""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT / "third_party" / "ucod_mkd_student"
sys.path.insert(0, str(REFERENCE_ROOT))
from net import Net  # noqa: E402


class Config:
    def __init__(self, backbone_path=None):
        self.mode = "train"
        self.snapshot = None
        self.backbone_path = str(backbone_path) if backbone_path else None


class Stage4Dataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.records = [json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        stem = record["image"]
        image_path = next((self.root / "train" / "Images").glob(f"{stem}.*"))
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pseudo = cv2.imread(str(self.root / "train" / record["pseudo_label"]), cv2.IMREAD_GRAYSCALE)
        box_background = cv2.imread(str(self.root / "train" / record["reliable_background"]), cv2.IMREAD_GRAYSCALE)
        weight = cv2.imread(str(self.root / "train" / record["confidence_map"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        pseudo = cv2.resize(pseudo, (512, 512), interpolation=cv2.INTER_NEAREST)
        box_background = cv2.resize(box_background, (512, 512), interpolation=cv2.INTER_NEAREST)
        weight = cv2.resize(weight, (512, 512), interpolation=cv2.INTER_LINEAR)
        if random.random() < 0.5:
            image, pseudo, box_background, weight = (array[:, ::-1].copy() for array in (image, pseudo, box_background, weight))
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (
            torch.from_numpy(image).permute(2, 0, 1),
            torch.from_numpy(pseudo == 1).float(),
            torch.from_numpy(box_background == 2).float(),
            torch.from_numpy(weight.astype(np.float32) / 255.0),
            torch.tensor(record["tier_value"], dtype=torch.long),
        )


def logits_from_model(model, image):
    output = model(image)
    return output[0] if isinstance(output, tuple) else output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Stage_4_Data")
    parser.add_argument("--backbone-path", type=Path, default=None, help="Path to pvt_v2_b4.pth (optional but recommended).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Stage_4_Checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu for a functional smoke test.")
    dataset = Stage4Dataset(args.data_dir)
    if not len(dataset):
        raise RuntimeError("Stage4_Data is empty. Run Stage_4.py first.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    device = torch.device(args.device)
    model = Net(Config(args.backbone_path)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for image, pseudo, outer_background, pixel_weight, tier in loader:
            image, pseudo = image.to(device), pseudo.to(device)
            outer_background, pixel_weight, tier = outer_background.to(device), pixel_weight.to(device), tier.to(device)
            logits = logits_from_model(model, image).squeeze(1)
            probability = torch.sigmoid(logits)
            # Strong-view prediction for the legacy SKD branch. Colour noise preserves pixel alignment.
            augmented = torch.clamp(image + torch.randn_like(image) * 0.05, -3.0, 3.0)
            augmented_probability = torch.sigmoid(logits_from_model(model, augmented).squeeze(1))
            pseudo_bce = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
            background_bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(pseudo), reduction="none")
            losses = []
            for sample in range(image.shape[0]):
                if tier[sample] == 2:  # BeKD high-quality branch: CE + L1 + L2.
                    loss = (pseudo_bce[sample] * pixel_weight[sample]).mean()
                    loss = loss + F.l1_loss(probability[sample], pseudo[sample]) + F.mse_loss(probability[sample], pseudo[sample])
                elif tier[sample] == 1:  # Normal branch: weighted pseudo-label CE.
                    loss = 0.5 * (pseudo_bce[sample] * pixel_weight[sample]).mean()
                else:  # Low branch: self knowledge distillation; no pseudo-mask supervision.
                    loss = 0.1 * F.l1_loss(probability[sample], augmented_probability[sample].detach())
                # Refer/box_tos.py is intentionally not used: only bbox-exterior pixels are reliable background.
                loss = loss + (background_bce[sample] * outer_background[sample]).sum() / outer_background[sample].sum().clamp_min(1)
                losses.append(loss)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch:03d}/{args.epochs}: loss={total / len(loader):.5f}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, args.output_dir / f"epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    main()
