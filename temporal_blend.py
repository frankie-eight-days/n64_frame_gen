"""Flow-guided temporal blending with TAA-style clamping and Z-buffer occlusion."""

import cv2
import numpy as np


class TemporalBlender:
    """Optical-flow-warped temporal blending with occlusion detection."""

    def __init__(self, width=320, height=240):
        self._width = width
        self._height = height
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        # Pre-compute remap base grids (never changes)
        self._grid_y, self._grid_x = np.mgrid[0:height, 0:width].astype(np.float32)

    def compute_flow(self, prev_gray, curr_gray):
        """Compute optical flow between two grayscale frames.

        Returns (H, W, 2) float32 flow field. Sub-1ms at 320x240 on CPU.
        """
        return self._dis.calc(prev_gray, curr_gray, None)

    def _get_grids(self, h, w):
        """Return base grids, rebuilding only when dimensions change."""
        if h != self._height or w != self._width:
            self._height, self._width = h, w
            self._grid_y, self._grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        return self._grid_x, self._grid_y

    def warp_frame(self, frame, flow):
        """Warp frame using flow field via cv2.remap."""
        h, w = flow.shape[:2]
        gx, gy = self._get_grids(h, w)
        map_x = gx + flow[:, :, 0]
        map_y = gy + flow[:, :, 1]
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def compute_occlusion_mask(self, prev_z14, curr_z14, flow, threshold=0.05):
        """Compare warped previous depth vs current depth.

        Returns float32 mask at z-buffer resolution where 0=visible, 1=occluded.
        """
        zh, zw = curr_z14.shape[:2]
        # Rescale flow to z-buffer resolution if needed
        if flow.shape[:2] != (zh, zw):
            sy = zh / flow.shape[0]
            sx = zw / flow.shape[1]
            zflow = cv2.resize(flow, (zw, zh))
            zflow[:, :, 0] *= sx
            zflow[:, :, 1] *= sy
        else:
            zflow = flow

        warped_depth = self.warp_frame(prev_z14, zflow)
        depth_range = max(curr_z14.max() - curr_z14.min(), 1.0)
        diff = np.abs(warped_depth - curr_z14) / depth_range
        mask = (diff > threshold).astype(np.float32)
        # Mark out-of-bounds pixels as occluded
        gx, gy = self._get_grids(zh, zw)
        map_x = gx + zflow[:, :, 0]
        map_y = gy + zflow[:, :, 1]
        oob = (map_x < 0) | (map_x >= zw) | (map_y < 0) | (map_y >= zh)
        mask[oob] = 1.0
        # Slight dilation to catch edges
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        return mask

    def neighborhood_clamp(self, warped, current, kernel_size=3):
        """Clamp warped history to min/max of current frame's local neighborhood (TAA-style)."""
        k = np.ones((kernel_size, kernel_size), np.uint8)
        lo = cv2.erode(current, k)   # per-channel min
        hi = cv2.dilate(current, k)  # per-channel max
        return np.clip(warped, lo, hi)

    def blend(self, new_frame, prev_stylized, flow, occlusion_mask,
              blend_strength=0.7):
        """Full warp-occlude-clamp-blend pipeline.

        Args:
            new_frame: Current diffusion output (H, W, 3) uint8
            prev_stylized: Previous blended output (H, W, 3) uint8
            flow: Optical flow field (H, W, 2) float32
            occlusion_mask: (H, W) float32, 0=visible, 1=occluded
            blend_strength: How much to favor warped history (0=all new, 1=all history)

        Returns:
            Blended frame (H, W, 3) uint8
        """
        warped = self.warp_frame(prev_stylized, flow)
        warped = self.neighborhood_clamp(warped, new_frame)
        # Expand mask to 3 channels
        mask_3ch = occlusion_mask[:, :, np.newaxis]
        # In occluded regions: use new diffusion output
        # In visible regions: weighted blend favoring warped history
        output = mask_3ch * new_frame.astype(np.float32) + (1 - mask_3ch) * (
            (1 - blend_strength) * new_frame.astype(np.float32) +
            blend_strength * warped.astype(np.float32)
        )
        return np.clip(output, 0, 255).astype(np.uint8)

    def visualize_flow(self, flow):
        """Convert flow field to HSV visualization for debug overlay."""
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
