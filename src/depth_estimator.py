import numpy as np
import torch
import cv2


class DepthEstimator:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    def _lazy_init(self):
        if self.model is not None:
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        self._lazy_init()
        input_batch = self.transform(frame_rgb).to(self.device)
        if self.device.type == "cuda":
            input_batch = input_batch.half()
        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = prediction.float().cpu().numpy()
        # Normalize to 0-255
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth = np.zeros_like(depth)
        depth = depth.astype(np.uint8)
        # Convert to 3-channel for ControlNet conditioning
        depth_3ch = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        return depth_3ch
