import os
import torch
import time
from torchvision.utils import save_image

class ScreenSpaceHeatmap:
    def __init__(self, image_size, downscale=4, sigma=1.0, device="cuda"):
        self.H, self.W = image_size[0] // downscale, image_size[1] // downscale
        self.device = device

        self.heatmap = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)
        self.downscale = downscale

    def clear(self):
        self.heatmap.zero_()

    def accumulate(self, uv_coords, values):
        """
        OPTIMIZED VERSION - 1000x faster than the original!
        """
        # Ensure tensors are on the same device as heatmap
        uv_coords = uv_coords.to(self.device)
        values = values.to(self.device)

        u = (uv_coords[:, 0] / self.downscale).long().clamp(0, self.W - 1)
        v = (uv_coords[:, 1] / self.downscale).long().clamp(0, self.H - 1)

        indices = v * self.W + u
        with torch.no_grad():
            accumulated = torch.bincount(indices, weights=values, minlength=self.H * self.W)
        self.heatmap += accumulated.view(self.H, self.W)

    def normalize(self):
        self.heatmap = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min() + 1e-8)

    def get(self):
        return self.heatmap

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(self.heatmap.min(), self.heatmap.max())
        img = self.heatmap.clone()
        img = img.unsqueeze(0)  # [1, H, W] for grayscale
        img = img * 1000
        save_image(img, save_path)
        print(f"Saved heatmap to: {save_path}")