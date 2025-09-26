import os
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from data import ResidualBinaryDataset, val_tfms_geometric, residualizer, _split_indices
from eval_balanced import _plot_cm
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_two_stage_full_val(
    dataset_root: str,
    binary_model: nn.Module,
    attr_model: nn.Module,
    attr_class_names: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 64,
    tta: bool = True,
    fake_threshold: float = 0.5,
):
    full = ResidualBinaryDataset(dataset_root, split="train")
    train_idx, val_idx = _split_indices(len(full), val_ratio, seed)

    class ValDataset(torch.utils.data.Dataset):
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs
            self.tf = val_tfms_geometric
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i):
            _, y_bin, p = self.base[self.idxs[i]]
            img = Image.open(p).convert('RGB')
            x = residualizer(self.tf(img))
            if y_bin == 0:
                y_e2e = 0
            else:
                rel = os.path.relpath(p, dataset_root)
                model_name = rel.split(os.sep)[0]
                if model_name in attr_class_names:
                    y_e2e = 1 + attr_class_names.index(model_name)
                else:
                    y_e2e = 0
            return x, y_bin, y_e2e

    val_ds = ValDataset(full, val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    y_true_bin, y_pred_bin = [], []
    y_true_attr, y_pred_attr = [], []

    binary_model.eval()
    attr_model.eval()

    with torch.no_grad():
        for x_batch, y_bin_batch, y_e2e_batch in val_loader:
            x_batch = x_batch.to(device)

            logits_bin = binary_model(x_batch)
            if tta:
                logits_bin = logits_bin + binary_model(torch.flip(x_batch, dims=[3]))

            probs_bin = torch.softmax(logits_bin, dim=1)
            pred_bin = (probs_bin[:, 1] >= fake_threshold).long()

            logits_attr = attr_model(x_batch)
            if tta:
                logits_attr = logits_attr + attr_model(torch.flip(x_batch, dims=[3]))
            pred_attr = logits_attr.argmax(dim=1)

            y_true_bin.extend(y_bin_batch.tolist())
            y_pred_bin.extend(pred_bin.cpu().tolist())

            fake_mask = (y_bin_batch == 1) & (pred_bin.cpu() == 1)
            if fake_mask.any():
                true_attr = (y_e2e_batch[fake_mask] - 1).clamp(min=0)
                y_true_attr.extend(true_attr.tolist())
                y_pred_attr.extend(pred_attr[fake_mask].cpu().tolist())

    print("\n========== Stage 1: Binary Classification ==========")
    cm1 = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm1)
    _plot_cm(cm1, ["real", "fake"], title="Stage 1: Real vs Fake")

    print("\nClassification Report:")
    print(classification_report(y_true_bin, y_pred_bin,
                              target_names=["real", "fake"], digits=3))

    if len(y_true_attr) > 0:
        print("\n========== Stage 2: Attribution ==========")
        cm2 = confusion_matrix(y_true_attr, y_pred_attr,
                              labels=list(range(len(attr_class_names))))
        print("Confusion Matrix:")
        print(cm2)
        _plot_cm(cm2, attr_class_names, title="Stage 2: Model Attribution")

        print("\nClassification Report: ")
        print(classification_report(y_true_attr, y_pred_attr,
                                  target_names=attr_class_names, digits=3))
    else:
        cm2 = None
        print("\n[Stage 2] No fake samples correctly identified by Stage 1")

    return {"cm_binary": cm1, "cm_attr": cm2}
