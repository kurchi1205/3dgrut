import torch
import torchvision
import torchvision.transforms.functional as TF
import os

def project_to_screen_space(points_3d, intrinsics, extrinsics):
    """
    Projects 3D world-space points into 2D screen-space pixel coordinates.

    Args:
        points_3d: [N, 3] tensor of 3D positions in world coordinates.
        intrinsics: [3, 3] tensor of camera intrinsics (fx, fy, cx, cy).
        extrinsics: [4, 4] tensor of world-to-camera transformation.

    Returns:
        uv: [N, 2] tensor of screen-space (u, v) pixel coordinates.
    """
    N = points_3d.shape[0]

    # Convert to homogeneous coordinates [x, y, z, 1]
    ones = torch.ones((N, 1), device=points_3d.device)
    points_h = torch.cat([points_3d, ones], dim=1)  # [N, 4]

    # Transform points from world to camera coordinates
    cam_points = (extrinsics @ points_h.T).T[:, :3]  # [N, 3]

    # Extract focal lengths and principal point
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Perspective divide (pinhole projection)
    X, Y, Z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    x_proj = X / Z
    y_proj = Y / Z

    # Convert to pixel coordinates
    u = fx * x_proj + cx
    v = fy * y_proj + cy

    # Stack to form final 2D screen-space coordinates
    uv = torch.stack([u, v], dim=1)  # [N, 2]
    return uv



def save_tensor_image(img_tensor, save_path):
    """
    Save a [1, H, W, 3] or [H, W, 3] tensor image to disk as a PNG.

    Args:
        img_tensor: Float tensor with values in [0, 255] or [0, 1].
        save_path: Output path (e.g., 'output/image_001.png')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Remove batch dim if present
    if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor[0]  # shape now [H, W, 3]

    # Ensure shape is [3, H, W] for save_image
    img_tensor = img_tensor.permute(2, 0, 1)  # [3, H, W]

    # If it's in [0, 255] range, scale to [0,1]
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    torchvision.utils.save_image(img_tensor, save_path)
    print(f"Saved image to {save_path}")
