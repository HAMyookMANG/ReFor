import os
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from residual import ToResidualTensor

IMG_SIZE = 224
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

train_tfms_geometric = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
])

val_tfms_geometric = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
])

residualizer = ToResidualTensor(ksize=5, sigma=1.2)

def _list_images(d):
    return [os.path.join(d, f) for f in os.listdir(d)
            if os.path.isfile(os.path.join(d,f)) and f.lower().endswith(IMG_EXTS)]

class ResidualBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.samples = []
        model_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for m in model_dirs:
            real_dir = os.path.join(root, m, "0_real")
            fake_dir = os.path.join(root, m, "1_fake")
            if os.path.isdir(real_dir):
                self.samples += [(p, 0) for p in _list_images(real_dir)]
            if os.path.isdir(fake_dir):
                self.samples += [(p, 1) for p in _list_images(fake_dir)]
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        self.tf = train_tfms_geometric if split == "train" else val_tfms_geometric

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p,y = self.samples[idx]
        img = Image.open(p).convert("RGB"); img = self.tf(img)
        return img, y, p

class ResidualAttributionDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split
        all_model_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
        model_names = []
        for m in sorted(all_model_dirs):
            fake_dir = os.path.join(root, m, "1_fake")
            if os.path.isdir(fake_dir):
                imgs = _list_images(fake_dir)
                if len(imgs) > 0:
                    model_names.append(m)
        if len(model_names) == 0:
            raise RuntimeError(f"No valid model folders with 1_fake images under: {root}")
        self.model_names = model_names
        self.class_to_idx = {m:i for i,m in enumerate(self.model_names)}
        self.samples: List[Tuple[str,int]] = []
        for m in self.model_names:
            fake_dir = os.path.join(root, m, "1_fake")
            for p in _list_images(fake_dir):
                self.samples.append((p, self.class_to_idx[m]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No fake images found under: {root}")
        self.transform = train_tfms_geometric if split == "train" else val_tfms_geometric

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, path

def _split_indices(n: int, val_ratio: float = 0.2, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(len(indices)*val_ratio))
    return indices[n_val:], indices[:n_val]

def _make_loader_with_residual(dataset, batch_size=32, num_workers=2, use_sampler=True):
    def _collate_with_residual(batch):
        imgs, labels, _paths = zip(*batch)
        tensors = [residualizer(img) for img in imgs]
        x = torch.stack(tensors, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y
    if use_sampler:
        labels = [dataset[i][1] for i in range(len(dataset))]
        K = max(labels)+1
        counts = [1]*K
        for y in labels: counts[y] += 1
        class_w = torch.tensor([1.0/c for c in counts], dtype=torch.float32)
        sample_w = [class_w[y].item() for y in labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True,
                          collate_fn=_collate_with_residual)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          collate_fn=_collate_with_residual)

def make_binary_loaders(dataset_root: str, batch_size=32, num_workers=2, val_ratio=0.2, seed=42):
    all_samples = []
    model_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    for m in model_dirs:
        real_dir = os.path.join(dataset_root, m, "0_real")
        fake_dir = os.path.join(dataset_root, m, "1_fake")
        if os.path.isdir(real_dir):
            all_samples += [(p, 0, p) for p in _list_images(real_dir)]
        if os.path.isdir(fake_dir):
            all_samples += [(p, 1, p) for p in _list_images(fake_dir)]
    tr_idx, va_idx = _split_indices(len(all_samples), val_ratio, seed)
    tr_samples = [all_samples[i] for i in tr_idx]
    va_samples = [all_samples[i] for i in va_idx]

    class BinarySubset(torch.utils.data.Dataset):
        def __init__(self, samples, split):
            self.samples = samples
            self.tf = train_tfms_geometric if split == "train" else val_tfms_geometric
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            path, label, _ = self.samples[i]
            img = Image.open(path).convert("RGB")
            img = self.tf(img)
            return img, label, path

    tr_ds = BinarySubset(tr_samples, "train")
    va_ds = BinarySubset(va_samples, "val")

    def _collate(batch):
        imgs, labels, _ = zip(*batch)
        x = torch.stack([residualizer(im) for im in imgs], 0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    labels = [y for (_, y, _) in tr_samples]
    K = max(labels) + 1
    counts = [1] * K
    for y in labels: counts[y] += 1
    class_w = torch.tensor([1.0/c for c in counts])
    sample_w = [class_w[y].item() for y in labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                           num_workers=num_workers, pin_memory=True, collate_fn=_collate)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, collate_fn=_collate)
    print(f"[Binary Data] train={len(tr_samples)} val={len(va_samples)}")
    return tr_loader, va_loader

def make_attr_loaders(dataset_root: str, batch_size=32, num_workers=2, val_ratio=0.2, seed=42):
    full = ResidualAttributionDataset(dataset_root, split="train")
    class_names = full.model_names
    train_idx, val_idx = _split_indices(len(full), val_ratio, seed)

    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, idxs, split):
            self.base = base; self.idxs = idxs
            self.tf = train_tfms_geometric if split=="train" else val_tfms_geometric
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i):
            _, y, p = self.base[self.idxs[i]]
            img = Image.open(p).convert('RGB')
            img = self.tf(img)
            return img, y, p

    train_ds = Subset(full, train_idx, "train")
    val_ds   = Subset(full, val_idx,   "val")

    def _collate_with_residual(batch):
        imgs, labels, _paths = zip(*batch)
        tensors = [residualizer(img) for img in imgs]
        x = torch.stack(tensors, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=_collate_with_residual)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=_collate_with_residual)
    print(f"[Attribution Data] classes={class_names}")
    print(f"[Attribution Data] train={len(train_idx)} val={len(val_idx)}")
    return train_loader, val_loader, class_names
