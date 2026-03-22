from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


DL_DIR = Path(__file__).resolve().parent
APP_DIR = DL_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

CONFIG_PATH = DL_DIR / "config.json"
EXPORTS_DIR = DL_DIR / "exports"
MODELS_DIR = DL_DIR / "models"
DATASET_JSONL_PATH = EXPORTS_DIR / "dl_dataset.jsonl"
MODEL_PATH = MODELS_DIR / "dl_residual_model.pt"
MODEL_META_PATH = MODELS_DIR / "dl_residual_model_meta.json"
OUTPUT_NAMES = ["composition", "light", "color", "technical", "subject", "impact", "total"]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_records() -> list[dict]:
    if not DATASET_JSONL_PATH.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {DATASET_JSONL_PATH}")
    records = []
    with DATASET_JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def import_torch():
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        return torch, Image, nn, Dataset, DataLoader, transforms
    except ImportError as error:
        raise SystemExit(
            "PyTorch 系の依存が見つかりません。"
            " 先に `pip install -r photo-evaluator-training-lab-app/dl-lab/requirements-dl.txt` を実行してください。"
        ) from error


def split_records(records: list[dict], validation_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * validation_split)) if len(shuffled) > 1 else 0
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:] if val_size else shuffled
    if not train_records:
        train_records, val_records = shuffled, []
    return train_records, val_records


def _read_targets(row: dict) -> list[float]:
    targets = row.get("targets") or {}
    values = []
    for name in OUTPUT_NAMES:
        value = targets.get(name)
        if value in ("", None):
            raise ValueError(f"target {name} is missing")
        values.append(float(value))
    return values


def train_model() -> dict:
    config = load_config()
    records = load_records()
    minimum_samples = int(config.get("minimum_samples") or 20)
    if len(records) < minimum_samples:
        raise SystemExit(f"学習用サンプルが不足しています: {len(records)}件 (必要: {minimum_samples}件)")

    torch, Image, nn, Dataset, DataLoader, transforms = import_torch()

    image_size = int(config.get("image_size") or 224)
    batch_size = int(config.get("batch_size") or 8)
    epochs = int(config.get("epochs") or 12)
    learning_rate = float(config.get("learning_rate") or 0.001)
    validation_split = float(config.get("validation_split") or 0.2)
    seed = int(config.get("seed") or 42)

    train_records, val_records = split_records(records, validation_split, seed)
    random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    class PhotoScoreDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(row["image_path"]).convert("RGB")
            tensor = transform(image)
            target = torch.tensor(_read_targets(row), dtype=torch.float32)
            return tensor, target

    class TinyScoreCNN(nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 96),
                nn.ReLU(inplace=True),
                nn.Linear(96, output_dim),
            )

        def forward(self, x):
            return self.head(self.features(x))

    train_loader = DataLoader(PhotoScoreDataset(train_records), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PhotoScoreDataset(val_records), batch_size=batch_size, shuffle=False) if val_records else None

    model = TinyScoreCNN(len(OUTPUT_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, targets in train_loader:
          optimizer.zero_grad()
          predictions = model(images)
          loss = criterion(predictions, targets)
          loss.backward()
          optimizer.step()
          batch_size_now = images.size(0)
          running_loss += loss.item() * batch_size_now
          sample_count += batch_size_now
        train_losses.append(running_loss / max(1, sample_count))

    validation_mae = None
    validation_mae_by_output = {}
    if val_loader:
        model.eval()
        mae_sum = torch.zeros(len(OUTPUT_NAMES), dtype=torch.float32)
        mae_count = 0
        with torch.no_grad():
            for images, targets in val_loader:
                predictions = model(images)
                mae_sum += torch.abs(predictions - targets).sum(dim=0)
                mae_count += targets.size(0)
        mae_values = (mae_sum / max(1, mae_count)).tolist()
        validation_mae_by_output = {
            name: round(float(value), 4)
            for name, value in zip(OUTPUT_NAMES, mae_values)
        }
        validation_mae = validation_mae_by_output.get("total")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "output_names": OUTPUT_NAMES,
        },
        MODEL_PATH,
    )

    metadata = {
        "model_type": "tiny_cnn_multi_score_regression_v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_JSONL_PATH),
        "sample_count": len(records),
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "target": "multi_score",
        "output_names": OUTPUT_NAMES,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_loss_final": train_losses[-1] if train_losses else None,
        "validation_mae": validation_mae,
        "validation_mae_by_output": validation_mae_by_output,
        "model_path": str(MODEL_PATH),
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = train_model()
    print("DL model training completed.")
    print(f"sample_count: {metadata['sample_count']}")
    print(f"train_count: {metadata['train_count']}")
    print(f"validation_count: {metadata['validation_count']}")
    print(f"train_loss_final: {metadata['train_loss_final']}")
    print(f"validation_mae: {metadata['validation_mae']}")
    print(f"output_names: {', '.join(metadata['output_names'])}")
    print(f"model_path: {metadata['model_path']}")


if __name__ == "__main__":
    main()
