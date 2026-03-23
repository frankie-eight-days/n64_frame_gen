"""Flow-warped noise for temporally coherent diffusion.

Warps the initial noise tensor using optical flow so that the noise pattern
moves with the scene, reducing texture crawl and shimmer in static regions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def warp_noise(prev_noise: torch.Tensor, flow: np.ndarray,
               vae_scale_factor: int = 8) -> torch.Tensor:
    """Warp a latent noise tensor using pixel-space optical flow.

    Args:
        prev_noise: (B, C, latent_h, latent_w) noise tensor
        flow: (H, W, 2) numpy array in pixel space (from optical flow)
        vae_scale_factor: ratio between pixel and latent dimensions (default 8)

    Returns:
        Warped noise tensor, same shape as prev_noise
    """
    h, w = prev_noise.shape[2], prev_noise.shape[3]

    # Scale flow from pixel space to latent space
    flow_latent = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / vae_scale_factor

    # Build normalized sampling grid (-1 to 1)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=prev_noise.device, dtype=prev_noise.dtype),
        torch.linspace(-1, 1, w, device=prev_noise.device, dtype=prev_noise.dtype),
        indexing='ij'
    )

    # Convert flow displacement to normalized coordinates and add to grid
    flow_tensor = torch.from_numpy(flow_latent).to(prev_noise.device, prev_noise.dtype)
    grid_x = grid_x + flow_tensor[:, :, 0] * 2.0 / w
    grid_y = grid_y + flow_tensor[:, :, 1] * 2.0 / h

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    # Expand grid for batch dimension if needed
    if prev_noise.shape[0] > 1:
        grid = grid.expand(prev_noise.shape[0], -1, -1, -1)

    warped = F.grid_sample(
        prev_noise, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # Re-normalize to maintain noise statistics (grid_sample can change variance)
    # Only re-normalize if the warped noise has significantly different stats
    orig_std = prev_noise.std()
    warped_std = warped.std()
    if warped_std > 0:
        warped = warped * (orig_std / warped_std)

    return warped
