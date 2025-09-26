from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class ToResidualTensor:
    def __init__(self, ksize=5, sigma=1.2, eps=1e-6):
        self.ksize = ksize
        self.sigma = sigma
        self.eps = eps
        ax = torch.arange(ksize) - (ksize-1)/2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma*sigma))
        kernel = kernel / kernel.sum()
        self.registered = False
        self.kernel = kernel

    def _register(self, device):
        k = self.kernel.to(device=device, dtype=torch.float32)
        self.weight = k.view(1,1,self.ksize,self.ksize)
        self.registered = True

    def __call__(self, img: Image.Image):
        x = transforms.functional.to_tensor(img)
        if x.shape[0] == 3:
            y = 0.299*x[0] + 0.587*x[1] + 0.114*x[2]
        else:
            y = x[0]
        y = y.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        if not self.registered:
            self._register(y.device)
        y_blur = F.conv2d(y, self.weight, padding=self.ksize//2)
        r = y - y_blur
        mean = r.mean()
        std  = r.std()
        r = (r - mean) / (std + self.eps)
        r3 = r.repeat(1,3,1,1).squeeze(0)  # (3,H,W)
        return r3
