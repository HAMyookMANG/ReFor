import sys, subprocess

def _pip_install(pkgs):
    for p in pkgs:
        print(f"[bootstrap] installing: {p}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + p.split())

def ensure_deps():
    _pip_install([
        "--upgrade pip setuptools wheel",
        "--upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121",
        "scikit-learn pillow matplotlib gdown==4.6.0",
    ])

def import_common():
    import os, math, random, io, itertools, json, warnings, hashlib, shutil
    from typing import List, Tuple, Optional, Dict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms, models

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    # Return a dict if someone wants to use it programmatically
    return {
        "os": os, "math": math, "random": random, "io": io, "itertools": itertools, "json": json,
        "warnings": warnings, "hashlib": hashlib, "Path": Path, "np": np, "pd": pd, "Image": Image,
        "torch": torch, "nn": nn, "F": F, "DataLoader": DataLoader, "WeightedRandomSampler": WeightedRandomSampler,
        "transforms": transforms, "models": models, "plt": plt,
        "confusion_matrix": confusion_matrix, "classification_report": classification_report,
        "shutil": shutil,
    }

if __name__ == "__main__":
    ensure_deps()
    print("[bootstrap] dependencies ensured. You can now import common ML libs.")
