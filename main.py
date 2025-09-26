import os
import torch
from data import make_binary_loaders, make_attr_loaders
from models import ResNetHead
from train_utils import fit_model, device
from e2e_eval import evaluate_two_stage_full_val

if __name__ == "__main__":
    BASE_DATASET_ROOT = "/content/prnu_dataset/data"
    VAL_RATIO = 0.2
    SEED = 42

    os.makedirs(os.path.join(BASE_DATASET_ROOT, "_ckpts_multi"), exist_ok=True)
    binary_ckpt_path = os.path.join(BASE_DATASET_ROOT, "_ckpts_multi", "binary_full_best.pth")
    attr_ckpt_path   = os.path.join(BASE_DATASET_ROOT, "_ckpts_multi", "attr_full_best.pth")

    print("="*60)
    print("Training Stage 1: Binary Classification")
    print("="*60)

    train_loader, val_loader = make_binary_loaders(
        BASE_DATASET_ROOT, batch_size=32, num_workers=2, val_ratio=VAL_RATIO, seed=SEED
    )
    binary_model = ResNetHead(num_classes=2).to(device)

    if os.path.exists(binary_ckpt_path):
        print(f"Loading existing binary checkpoint from {binary_ckpt_path}")
        binary_model.load_state_dict(torch.load(binary_ckpt_path, map_location=device))
        binary_model.eval()
    else:
        best_acc_binary = fit_model(
            binary_model, train_loader, val_loader,
            epochs=20, lr=1e-3, wd=1e-4,
            save_path=binary_ckpt_path,
            class_names=["real", "fake"]
        )
        print(f"[Binary] Training completed with accuracy: {best_acc_binary:.4f}")

    print("\n" + "="*60)
    print("Training Stage 2: Attribution Classification")
    print("="*60)

    attr_train_loader, attr_val_loader, attr_classes = make_attr_loaders(
        BASE_DATASET_ROOT, batch_size=32, num_workers=2, val_ratio=VAL_RATIO, seed=SEED
    )
    print(f"Attribution classes: {attr_classes}")
    attr_model = ResNetHead(num_classes=len(attr_classes)).to(device)

    if os.path.exists(attr_ckpt_path):
        print(f"Loading existing attribution checkpoint from {attr_ckpt_path}")
        attr_model.load_state_dict(torch.load(attr_ckpt_path, map_location=device))
        attr_model.eval()
    else:
        best_acc_attr = fit_model(
            attr_model, attr_train_loader, attr_val_loader,
            epochs=20, lr=1e-3, wd=1e-4,
            save_path=attr_ckpt_path,
            class_names=attr_classes
        )
        print(f"[Attribution] Training completed with accuracy: {best_acc_attr:.4f}")

    print("\n" + "="*60)
    print("Evaluating Two-Stage Pipeline on Val Split")
    print("="*60)

    results = evaluate_two_stage_full_val(
        dataset_root=BASE_DATASET_ROOT,
        binary_model=binary_model,
        attr_model=attr_model,
        attr_class_names=attr_classes,
        val_ratio=VAL_RATIO,
        seed=SEED,
        batch_size=64,
        tta=True,
        fake_threshold=0.5
    )

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
