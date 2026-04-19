"""Quick script to evaluate a saved checkpoint on test set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from reproduce_shu import SHUDataset, SHUModel, collate_fn


def main():
    device = torch.device("cuda:0")
    ckpt = "/home/zekun/projects/0-jobs/SigmaNova/results/shu_mi/epoch5_acc_0.61443_pr_0.67553_roc_0.67487.pth"

    model = SHUModel(
        pretrained_path=None,
        classifier_type="all_patch_reps",
        dropout=0.1,
        device=device,
    )
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt}")

    test_set = SHUDataset(
        "/home/zekun/projects/0-jobs/SigmaNova/data/processed", mode="test"
    )
    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    truths, preds, scores = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

    truths, preds, scores = np.array(truths), np.array(preds), np.array(scores)
    acc = balanced_accuracy_score(truths, preds)
    roc = roc_auc_score(truths, scores)
    prec, rec, _ = precision_recall_curve(truths, scores, pos_label=1)
    pr = auc(rec, prec)
    cm = confusion_matrix(truths, preds)
    print(f"Test Balanced Accuracy: {acc:.5f}")
    print(f"Test PR-AUC:           {pr:.5f}")
    print(f"Test ROC-AUC:          {roc:.5f}")
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
