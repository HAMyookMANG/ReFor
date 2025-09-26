from typing import List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from data import residualizer, val_tfms_geometric
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def _softmax_np(logits: torch.Tensor):
    p = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return p

@torch.no_grad()
def predict_two_stage(
    img_path: str,
    binary_model: nn.Module,
    attr_model: nn.Module,
    attr_class_names: List[str],
    tta: bool = True,
    fake_threshold: float = 0.5
):
    img = Image.open(img_path).convert('RGB')
    xg = val_tfms_geometric(img)
    xr = residualizer(xg).unsqueeze(0).to(device)

    binary_model.eval()
    with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
        logit_b = binary_model(xr)
        if tta: logit_b = logit_b + binary_model(torch.flip(xr, dims=[3]))
    prob_fake = float(_softmax_np(logit_b)[1])  # [real, fake]
    stage1_pred = 'fake' if prob_fake >= fake_threshold else 'real'

    stage2 = {'pred_model': None, 'prob': None, 'probs': None}
    if stage1_pred == 'fake':
        attr_model.eval()
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logit_a = attr_model(xr)
            if tta: logit_a = logit_a + attr_model(torch.flip(xr, dims=[3]))
        probs = _softmax_np(logit_a)
        idx = int(np.argmax(probs))
        stage2 = {'pred_model': attr_class_names[idx], 'prob': float(probs[idx]), 'probs': probs.tolist()}

    return {
        'stage1': {'pred': stage1_pred, 'prob_fake': prob_fake},
        'stage2': stage2
    }
