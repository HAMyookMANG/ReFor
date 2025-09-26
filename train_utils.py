import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def _plot_cm(cm, class_names, title="Confusion Matrix", normalize=True):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True', xlabel='Predicted', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

def _print_report_and_cm(y_true, y_pred, class_names=None, title_prefix=""):
    if class_names is None:
        class_names = ["real", "fake"] if len(set(y_true)) <= 2 else [str(i) for i in sorted(set(y_true))]
    print(f"\n{title_prefix}Classification Report")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print(f"{title_prefix}Confusion Matrix (counts)")
    print(cm)
    _plot_cm(cm, class_names, title=f"{title_prefix}Confusion Matrix (normalized)")

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    loss_m = acc_m = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc = accuracy_from_logits(logits, y)
        loss_m += loss.item()
        acc_m += acc
    n = len(loader)
    return loss_m/n, acc_m/n

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_m = acc_m = 0.0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logits = model(x)
            loss = criterion(logits, y)
        acc = accuracy_from_logits(logits, y)
        loss_m += loss.item()
        acc_m += acc
        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    n = len(loader)
    return (loss_m/n, acc_m/n, np.array(all_true), np.array(all_pred))

def fit_model(model, train_loader, val_loader, epochs=8, lr=1e-3, wd=1e-4,
              head_freeze=True, save_path="/content/best.pth", label_smoothing=0.1,
              class_names=None, print_metrics=True):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if head_freeze:
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        for p in model.backbone.layer4.parameters(): p.requires_grad = False
        for p in model.backbone.fc.parameters():     p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

    tr_losses=[]; va_losses=[]; tr_accs=[]; va_accs=[]
    best_acc=-1.0
    head_epochs = max(3, min(5, epochs//2))
    print(f"[Training] Starting training with {epochs} epochs (head: {head_epochs}, finetune: {epochs-head_epochs})")

    for ep in range(1, head_epochs+1):
        tl, ta = train_one_epoch(model, train_loader, opt, scaler, criterion)
        vl, va, y_true, y_pred = evaluate(model, val_loader, criterion)
        scheduler.step()
        tr_losses.append(tl); tr_accs.append(ta); va_losses.append(vl); va_accs.append(va)
        print(f"[Head {ep}/{head_epochs}] train {tl:.4f}/{ta:.3f} | val {vl:.4f}/{va:.3f}")
        if va > best_acc:
            best_acc = va
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("  -> saved best (head stage)")

    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=max(lr*0.03, 3e-5), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs-head_epochs))

    for ep in range(head_epochs+1, epochs+1):
        tl, ta = train_one_epoch(model, train_loader, opt, scaler, criterion)
        vl, va, y_true, y_pred = evaluate(model, val_loader, criterion)
        scheduler.step()
        tr_losses.append(tl); tr_accs.append(ta); va_losses.append(vl); va_accs.append(va)
        print(f"[Finetune {ep}/{epochs}] train {tl:.4f}/{ta:.3f} | val {vl:.4f}/{va:.3f}")
        if va > best_acc:
            best_acc = va
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("  -> saved best")

    import matplotlib.pyplot as plt
    epochs_range = range(1, len(tr_losses)+1)
    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, tr_losses, marker='o', label='Train Loss')
    plt.plot(epochs_range, va_losses, marker='s', label='Val Loss')
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.legend(); plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, tr_accs, marker='o', label='Train Acc')
    plt.plot(epochs_range, va_accs, marker='s', label='Val Acc')
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.show()

    if os.path.exists(save_path):
        state = torch.load(save_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()

    print(f"[Best Val Acc] {best_acc:.3f} (saved to {save_path})")

    if val_loader is not None and print_metrics:
        vl, va, y_true, y_pred = evaluate(model, val_loader, criterion)
        print(f"[Final Eval @ Best CKPT] val loss/acc: {vl:.4f}/{va:.3f}")
        _print_report_and_cm(y_true, y_pred, class_names=class_names, title_prefix="[Best CKPT] ")

    return best_acc
