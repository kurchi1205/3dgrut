import os
import torch
from torchvision.utils import save_image

class ScreenSpaceHeatmap:
    def __init__(self, image_size, downscale=4, sigma=1.0):
        self.H, self.W = image_size[0] // downscale, image_size[1] // downscale
        self.heatmap = torch.zeros((self.H, self.W), dtype=torch.float32)
        self.downscale = downscale

    def clear(self):
        self.heatmap.zero_()

    def accumulate(self, uv_coords, values):
        """
        uv_coords: [N, 2] screen-space positions in pixel coords (float)
        values: [N] importance scores (e.g., gradient norms)
        """
        u = (uv_coords[:, 0] / self.downscale).long().clamp(0, self.W - 1)
        v = (uv_coords[:, 1] / self.downscale).long().clamp(0, self.H - 1)

        for i in range(u.shape[0]):
            self.heatmap[v[i], u[i]] += values[i].item()

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