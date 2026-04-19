"""
Reproduce CBraMod paper results on SHU-MI dataset.

This script finetunes the pretrained CBraMod backbone on the SHU Motor Imagery
binary classification task, with wandb logging.

Paper reference metrics for SHU-MI:
  - Balanced Accuracy, ROC-AUC, PR-AUC

Usage:
  .venv/bin/python reproduce_shu.py
"""

import argparse
import copy
import os
import pickle
import random
from pathlib import Path
from timeit import default_timer as timer

import lmdb
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# wandb setup
# ---------------------------------------------------------------------------
import wandb

WANDB_API_KEY = "WANDB_API_KEY_REDACTED"


# ===========================================================================
# Model (copied from authors, kept faithful to reproduce)
# ===========================================================================
# We import directly from the repo modules so the architecture is identical.
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.cbramod import CBraMod  # noqa: E402


class SHUModel(nn.Module):
    """Downstream binary-classification model for SHU-MI."""

    def __init__(
        self,
        pretrained_path: str | None,
        classifier_type: str = "all_patch_reps",
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        if pretrained_path:
            state_dict = torch.load(
                pretrained_path, map_location=device, weights_only=True
            )
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded pretrained backbone from {pretrained_path}")
        # Remove the pretraining projection head
        self.backbone.proj_out = nn.Identity()

        # SHU-MI: 32 channels × 4 patches × 200 d_model
        flat_dim = 32 * 4 * 200
        if classifier_type == "avgpooling_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier_type == "all_patch_reps_onelayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat_dim, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier_type == "all_patch_reps_twolayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat_dim, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier_type == "all_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat_dim, 4 * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


# ===========================================================================
# Dataset
# ===========================================================================
class SHUDataset(Dataset):
    """SHU-MI dataset backed by LMDB.

    A shared LMDB environment is used across all splits to avoid the
    'already open in this process' error.
    """

    _shared_db: lmdb.Environment | None = None
    _shared_path: str | None = None

    @classmethod
    def _get_db(cls, data_dir: str) -> lmdb.Environment:
        if cls._shared_db is None or cls._shared_path != data_dir:
            cls._shared_db = lmdb.open(
                data_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            cls._shared_path = data_dir
        return cls._shared_db

    def __init__(self, data_dir: str, mode: str = "train", augment: bool = False):
        super().__init__()
        self.augment = augment
        self.db = self._get_db(data_dir)
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get("__keys__".encode()))
            self.keys = all_keys[mode]
        print(f"SHUDataset[{mode}]: {len(self.keys)} samples")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair["sample"]
        label = pair["label"]
        data = data / 100.0  # same normalization as authors
        if getattr(self, "augment", False):
            # Channel dropout (p=0.1, applied 50% of the time)
            if np.random.rand() < 0.5:
                channel_mask = np.random.rand(data.shape[0]) > 0.1
                data = data * channel_mask[:, None, None]
            # Gaussian noise (std=0.02, applied 50% of the time)
            if np.random.rand() < 0.5:
                data += np.random.normal(0, 0.02, size=data.shape).astype(np.float32)

        return data, label


def collate_fn(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_data = np.array([x[0] for x in batch])
    y_label = np.array([x[1] for x in batch])
    return torch.from_numpy(x_data).float(), torch.from_numpy(y_label).float()


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate_binary(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> dict:
    """Return balanced accuracy, PR-AUC, ROC-AUC and confusion matrix."""
    model.eval()
    truths, preds, scores = [], [], []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        score_y = torch.sigmoid(pred)
        pred_y = torch.gt(score_y, 0.5).long()
        truths += y.long().cpu().squeeze().numpy().tolist()
        preds += pred_y.cpu().squeeze().numpy().tolist()
        scores += score_y.cpu().numpy().tolist()

    truths = np.array(truths)
    preds = np.array(preds)
    scores = np.array(scores)
    balanced_acc = balanced_accuracy_score(truths, preds)
    roc = roc_auc_score(truths, scores)
    precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
    pr = auc(recall, precision)
    cm = confusion_matrix(truths, preds)
    return {"balanced_acc": balanced_acc, "pr_auc": pr, "roc_auc": roc, "cm": cm}


# ===========================================================================
# Training
# ===========================================================================
def train(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- data ----------
    train_set = SHUDataset(args.datasets_dir, mode="train", augment=args.augment)
    val_set = SHUDataset(args.datasets_dir, mode="val", augment=False)
    test_set = SHUDataset(args.datasets_dir, mode="test", augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,  # FIXED: no shuffle for val
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,  # FIXED: no shuffle for test
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # ---------- model ----------
    model = SHUModel(
        pretrained_path=args.foundation_dir if args.use_pretrained_weights else None,
        classifier_type=args.classifier,
        dropout=args.dropout,
        device=device,
    ).to(device)

    criterion = BCEWithLogitsLoss().to(device)

    # ---------- optimizer (multi-lr as in paper) ----------
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
            param.requires_grad = not args.frozen
        else:
            other_params.append(param)

    classifier_lr = 0.001 * (args.batch_size / 256) ** 0.5
    if args.multi_lr:
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr},
                {"params": other_params, "lr": classifier_lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    data_length = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * data_length, eta_min=1e-6
    )

    # ---------- wandb ----------
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(
        project="CBraMod-SHU-MI",
        name=f"shu_reproduce_seed{args.seed}_bs{args.batch_size}_lr{args.lr}",
        config=vars(args),
    )
    wandb.watch(model, log="gradients", log_freq=50)

    # ---------- training loop ----------
    best_roc_auc = 0.0
    best_model_states = None
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = timer()
        losses = []

        for x, y in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", mininterval=5
        ):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        lr_current = optimizer.param_groups[0]["lr"]
        elapsed = (timer() - start_time) / 60

        # validation
        val_metrics = evaluate_binary(model, val_loader, device)

        print(
            f"Epoch {epoch}: loss={mean_loss:.5f}, "
            f"val_balanced_acc={val_metrics['balanced_acc']:.5f}, "
            f"val_pr_auc={val_metrics['pr_auc']:.5f}, "
            f"val_roc_auc={val_metrics['roc_auc']:.5f}, "
            f"lr={lr_current:.6f}, "
            f"time={elapsed:.2f}min"
        )
        print(f"  Confusion matrix:\n{val_metrics['cm']}")

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": mean_loss,
                "train/lr": lr_current,
                "val/balanced_acc": val_metrics["balanced_acc"],
                "val/pr_auc": val_metrics["pr_auc"],
                "val/roc_auc": val_metrics["roc_auc"],
                "val/cm": wandb.Table(
                    data=val_metrics["cm"].tolist(),
                    columns=["pred_0", "pred_1"],
                ),
                "time/epoch_min": elapsed,
            }
        )

        if val_metrics["roc_auc"] > best_roc_auc:
            print(
                f"  >> New best ROC-AUC: {val_metrics['roc_auc']:.5f} (was {best_roc_auc:.5f})"
            )
            best_roc_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_model_states = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # ---------- test ----------
    if best_model_states is None:
        print("WARNING: No improvement during training, using last model state.")
        best_model_states = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_states)
    test_metrics = evaluate_binary(model, test_loader, device)

    print("=" * 60)
    print("TEST RESULTS (best val model from epoch {})".format(best_epoch))
    print("=" * 60)
    print(f"  Balanced Accuracy: {test_metrics['balanced_acc']:.5f}")
    print(f"  PR-AUC:            {test_metrics['pr_auc']:.5f}")
    print(f"  ROC-AUC:           {test_metrics['roc_auc']:.5f}")
    print(f"  Confusion Matrix:\n{test_metrics['cm']}")

    wandb.log(
        {
            "test/balanced_acc": test_metrics["balanced_acc"],
            "test/pr_auc": test_metrics["pr_auc"],
            "test/roc_auc": test_metrics["roc_auc"],
            "test/best_epoch": best_epoch,
        }
    )

    # save model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (
        f"epoch{best_epoch}_acc_{test_metrics['balanced_acc']:.5f}"
        f"_pr_{test_metrics['pr_auc']:.5f}"
        f"_roc_{test_metrics['roc_auc']:.5f}.pth"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    wandb.finish()


# ===========================================================================
# CLI
# ===========================================================================
def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description="CBraMod SHU-MI Reproduction")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--classifier",
        type=str,
        default="all_patch_reps",
        choices=[
            "all_patch_reps",
            "all_patch_reps_twolayer",
            "all_patch_reps_onelayer",
            "avgpooling_patch_reps",
        ],
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/home/zekun/projects/0-jobs/SigmaNova/data/processed",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/zekun/projects/0-jobs/SigmaNova/results/shu_mi",
    )
    parser.add_argument(
        "--foundation_dir",
        type=str,
        default="pretrained_weights/pretrained_weights.pth",
    )
    parser.add_argument("--use_pretrained_weights", action="store_true", default=True)
    parser.add_argument(
        "--no_pretrained_weights", dest="use_pretrained_weights", action="store_false"
    )
    parser.add_argument("--multi_lr", action="store_true", default=True)
    parser.add_argument("--no_multi_lr", dest="multi_lr", action="store_false")
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable EEG data augmentation",
    )

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
