"""
N64 DLSS Live - Real-time StreamDiffusion + TensorRT enhancement for N64 games

Usage:  python n64_dlss_live.py

Controls:
    F2          Toggle diffusion on/off
    ESC         Quit
    Arrow Keys  D-Pad
    WASD        Analog Stick
    X/Z         A/B Buttons
    Left Shift  Z Trigger
    Q/E         L/R Shoulder
    Enter       Start
    I/J/K/L     C-Buttons
    F1          Reset
"""

import os
import sys
import time
import threading
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk
import cv2
from ui_utils import ToolTip, set_widget_state, TOOLTIPS

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(SRC_DIR)  # repo root
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "StreamDiffusion"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "n64emulator"))

import torch
from utils.wrapper import StreamDiffusionWrapper
from temporal_blend import TemporalBlender
from sm64_state_reader import SM64StateReader

# ============================================================
# Presets
# ============================================================

PRESETS = {
    "Default Enhance": {
        "prompt": "high quality, enhanced, sharp, detailed, remastered",
        "strength": 0.35,
        "delta": 0.5,
    },
    "Studio Ghibli": {
        "prompt": "studio ghibli style, anime background, lush detailed",
        "strength": 0.50,
        "delta": 0.5,
    },
    "Watercolor": {
        "prompt": "watercolor painting, soft edges, flowing colors, artistic",
        "strength": 0.45,
        "delta": 0.5,
    },
    "Oil Painting": {
        "prompt": "oil painting, thick brushstrokes, impressionist, painterly",
        "strength": 0.45,
        "delta": 0.5,
    },
    "Synthwave": {
        "prompt": "neon synthwave, glowing edges, cyberpunk, purple blue",
        "strength": 0.55,
        "delta": 0.5,
    },
    "Modern Game": {
        "prompt": "modern AAA video game, ray traced, photorealistic, unreal engine 5",
        "strength": 0.35,
        "delta": 0.5,
    },
}

NUM_DENOISE_STEPS = 4


def strength_to_t_index_list(strength: float) -> list[int]:
    strength = max(0.0, min(1.0, strength))
    start = int(round(37 * (1.0 - strength)))  # 37 at min, 0 at max
    end = 44 + int(round(2 * (1.0 - strength)))  # ~44-46
    indices = []
    for i in range(NUM_DENOISE_STEPS):
        idx = start + int(round(i * (end - start) / (NUM_DENOISE_STEPS - 1)))
        indices.append(max(0, min(49, idx)))
    return indices

# ============================================================
# ControlNet helpers
# ============================================================

CONTROLNET_MODELS = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
}


def extract_canny(frame: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


SM64_ZBUFFER_ADDR = 0x00000400
SM64_ZBUFFER_SIZE = 320 * 240 * 2  # 153,600 bytes


def _decode_n64_z(rdram_array, address, width, height):
    """Decode N64 Z-buffer from RDRAM into a smooth float32 depth array."""
    size = width * height * 2
    raw = np.frombuffer(rdram_array, dtype=np.uint8, count=size, offset=address)
    # Undo mupen64plus/parallel_n64 32-bit word byte swap.
    # The emulator stores RDRAM with bytes reversed within each 4-byte word.
    raw_swapped = raw.reshape(-1, 4)[:, ::-1].flatten()
    # Now interpret as big-endian u16
    high = raw_swapped[0::2].astype(np.uint16)
    low = raw_swapped[1::2].astype(np.uint16)
    z16 = (high << 8) | low

    # The raw z16 value IS the non-linear encoding. Since the N64 uses a
    # piecewise-linear approximation of 1/z, we can simply treat the raw
    # 14-bit value (bits 15:2) as a monotonic depth index.  Higher = farther.
    z14 = (z16 >> 2).astype(np.float32)
    return z14.reshape(height, width), z16


def read_n64_zbuffer(rdram_array, address=SM64_ZBUFFER_ADDR, width=320, height=240):
    """Read N64 Z-buffer from RDRAM, decode, and return as (H,W,3) uint8 depth map."""
    z_float, _ = _decode_n64_z(rdram_array, address, width, height)

    # Normalize to 0-255
    z_min, z_max = z_float.min(), z_float.max()
    if z_max > z_min:
        depth = ((z_float - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)
    else:
        depth = np.zeros((height, width), dtype=np.uint8)

    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth = clahe.apply(depth)

    return cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)


def read_n64_zbuffer_with_depth(rdram_array, address=SM64_ZBUFFER_ADDR, width=320, height=240):
    """Read N64 Z-buffer and return both the RGB visualization and raw z14 float.

    Returns:
        (depth_rgb, z14_float) tuple where:
        - depth_rgb: (H, W, 3) uint8 CLAHE-enhanced depth visualization
        - z14_float: (H, W) float32 raw depth values for occlusion detection
    """
    z_float, z16 = _decode_n64_z(rdram_array, address, width, height)

    if z_float.max() == z_float.min():
        depth_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        return depth_rgb, z_float

    z_norm = (z_float - z_float.min()) / (z_float.max() - z_float.min())
    z_u8 = (z_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    z_clahe = clahe.apply(z_u8)
    depth_rgb = cv2.cvtColor(z_clahe, cv2.COLOR_GRAY2RGB)
    return depth_rgb, z_float


def dump_zbuffer_snapshot(rdram_array, raw_frame=None, address=SM64_ZBUFFER_ADDR,
                          width=320, height=240):
    """Save multiple Z-buffer visualizations to disk for analysis."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "zbuffer_dumps")
    os.makedirs(out_dir, exist_ok=True)

    z_float, z16 = _decode_n64_z(rdram_array, address, width, height)

    z_min, z_max = z_float.min(), z_float.max()

    # 1) Raw z14 normalized to 0-255 (smooth — no exponent banding)
    if z_max > z_min:
        depth_norm = ((z_float - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)
    else:
        depth_norm = np.zeros((height, width), dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_depth_raw.png"), depth_norm)

    # 2) CLAHE enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth_clahe = clahe.apply(depth_norm)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_depth_clahe.png"), depth_clahe)

    # 3) Histogram equalized
    depth_histeq = cv2.equalizeHist(depth_norm)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_depth_histeq.png"), depth_histeq)

    # 4) Colormap (turbo) for easy visualization
    depth_color = cv2.applyColorMap(depth_clahe, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_depth_turbo.png"), depth_color)

    # 5) Inverted (near=bright, far=dark — more intuitive)
    depth_inv = 255 - depth_clahe
    cv2.imwrite(os.path.join(out_dir, f"{ts}_depth_inverted.png"), depth_inv)

    # 6) Raw game frame if available
    if raw_frame is not None:
        cv2.imwrite(os.path.join(out_dir, f"{ts}_game_frame.png"),
                    cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))

    # 7) Stats text file
    z16_flat = z16.flatten()
    exponent = (z16_flat >> 13) & 0x7
    with open(os.path.join(out_dir, f"{ts}_stats.txt"), "w") as f:
        f.write(f"Z-Buffer Stats\n")
        f.write(f"==============\n")
        f.write(f"Raw z16 range: {z16_flat.min()} - {z16_flat.max()}\n")
        f.write(f"z14 (bits 15:2) range: {z_min:.0f} - {z_max:.0f}\n")
        unique_exp, exp_counts = np.unique(exponent, return_counts=True)
        f.write(f"\nExponent distribution:\n")
        for e, c in zip(unique_exp, exp_counts):
            pct = c / len(exponent) * 100
            f.write(f"  exp={e}: {c:>6d} pixels ({pct:5.1f}%)\n")
        f.write(f"\nz14 histogram (8 bins):\n")
        hist, edges = np.histogram(z_float.flatten(), bins=8)
        for i in range(len(hist)):
            f.write(f"  {edges[i]:>8.0f} - {edges[i+1]:>8.0f}: {hist[i]:>6d} pixels\n")

    print(f"Z-buffer snapshot saved to {out_dir}/{ts}_*")
    return out_dir


# ============================================================
# Diffusion Processor (background thread)
# ============================================================

class DiffusionProcessor:
    def __init__(self):
        self.stream = None
        self.enabled = False
        self.running = True

        self.prompt = PRESETS["Default Enhance"]["prompt"]
        self.strength = PRESETS["Default Enhance"]["strength"]
        self.delta = PRESETS["Default Enhance"]["delta"]

        self._pending = None
        self._pending_lock = threading.Lock()

        # Frame exchange (GIL makes ref assignment atomic)
        self.latest_raw_frame = None
        self.latest_enhanced_frame = None

        self.diff_fps = 0.0
        self.status = "Initializing..."

        # ControlNet settings
        self.controlnet_enabled = False
        self.controlnet_mode = "canny"  # "canny", "depth_midas", or "depth_zbuffer"
        self.controlnet_scale = 0.7
        self.canny_low = 80
        self.canny_high = 180
        self._depth_estimator = None
        self._controlnet_pending = None
        self._rdram = None
        self.latest_zbuffer_debug = None
        self.show_zbuffer_debug = False
        self._prev_enhanced = None  # Previous diffusion output for temporal blending
        self.temporal_blend = 0.3   # 0.0 = no blending (raw), 1.0 = freeze on previous frame

        # Temporal consistency (flow-guided blending + feature bank)
        self._temporal_blender = TemporalBlender(width=320, height=240)
        self._state_reader = None       # SM64StateReader (initialized when RDRAM available)
        self._prev_raw_gray = None      # Previous N64 frame (grayscale, for optical flow)
        self._prev_z14 = None           # Previous Z-buffer float (for occlusion)
        self.temporal_mode = "naive"     # "off", "naive", "flow", "v2v", "v2v+flow"
        self.flow_blend_strength = 0.7
        self.occlusion_threshold = 0.05
        self.use_warped_noise = False
        self.use_cached_attn = False
        self.feature_injection_strength = 0.8
        self.feature_similarity_threshold = 0.98
        self.show_flow_debug = False
        self.latest_flow_debug = None
        self.motion_source = "optical_flow"  # "optical_flow" or "game_state"

    def set_rdram(self, rdram_array):
        self._rdram = rdram_array
        if rdram_array is not None and self._state_reader is None:
            self._state_reader = SM64StateReader(rdram_array)

    def init_model(self):
        controlnet_model_id = None
        if self.controlnet_enabled:
            if self.controlnet_mode.startswith("depth_"):
                controlnet_model_id = CONTROLNET_MODELS["depth"]
            else:
                controlnet_model_id = CONTROLNET_MODELS.get(self.controlnet_mode)
            self.status = f"Loading StreamDiffusion + ControlNet ({self.controlnet_mode})..."
        else:
            self.status = "Loading StreamDiffusion + TensorRT..."
        print(self.status)

        if self.controlnet_enabled and self.controlnet_mode == "depth_midas" and self._depth_estimator is None:
            from depth_estimator import DepthEstimator
            self._depth_estimator = DepthEstimator()

        # TensorRT compiles the UNet into an opaque engine without attention
        # layer access, so V2V (cached attention) requires PyTorch fallback.
        accel = "none" if self.use_cached_attn else "tensorrt"

        self.stream = StreamDiffusionWrapper(
            model_id_or_path="Lykon/dreamshaper-7",
            t_index_list=strength_to_t_index_list(self.strength),
            mode="img2img",
            output_type="pil",
            frame_buffer_size=1,
            width=320,
            height=240,
            warmup=10,
            acceleration=accel,
            use_lcm_lora=True,
            use_tiny_vae=True,
            use_denoising_batch=True,
            cfg_type="self",
            seed=42,
            use_safety_checker=False,
            engine_dir=os.path.join(SCRIPT_DIR, "engines"),
            controlnet_model_id=controlnet_model_id,
            controlnet_scale=self.controlnet_scale,
            use_cached_attn=self.use_cached_attn,
            feature_injection_strength=self.feature_injection_strength,
            feature_similarity_threshold=self.feature_similarity_threshold,
        )
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt="blurry, low quality, distorted, artifacts, ugly",
            num_inference_steps=50,
            guidance_scale=2.0,
            delta=self.delta,
        )
        # Warmup
        dummy = Image.new("RGB", (320, 240))
        tensor = self.stream.preprocess_image(dummy)
        for _ in range(5):
            self.stream(image=tensor)
        torch.cuda.synchronize()
        self.status = "Ready (diffusion OFF)"
        print("StreamDiffusion ready!")

    def request_settings(self, prompt, strength, delta):
        with self._pending_lock:
            self._pending = {"prompt": prompt, "strength": strength, "delta": delta}

    def _apply_pending(self):
        with self._pending_lock:
            if self._pending is None:
                return
            s = self._pending
            self._pending = None

        self.status = "Applying settings..."
        self.prompt = s["prompt"]
        self.strength = s["strength"]
        self.delta = s["delta"]

        self.stream.stream.t_list = strength_to_t_index_list(self.strength)
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt="blurry, low quality, distorted, artifacts, ugly",
            num_inference_steps=50,
            guidance_scale=2.0,
            delta=self.delta,
        )
        self.status = "Running"

    def request_controlnet_settings(self, scale=None, canny_low=None, canny_high=None):
        if scale is not None:
            self.controlnet_scale = scale
            if self.stream and self.stream.stream.controlnet is not None:
                self.stream.stream.controlnet_scale = scale
        if canny_low is not None:
            self.canny_low = canny_low
        if canny_high is not None:
            self.canny_high = canny_high

    def request_reload(self, controlnet_enabled, controlnet_mode, use_cached_attn=None):
        with self._pending_lock:
            self._controlnet_pending = {
                "enabled": controlnet_enabled,
                "mode": controlnet_mode,
                "use_cached_attn": use_cached_attn,
            }

    def _apply_controlnet_reload(self):
        with self._pending_lock:
            if self._controlnet_pending is None:
                return False
            s = self._controlnet_pending
            self._controlnet_pending = None

        old_enabled = self.controlnet_enabled
        old_mode = self.controlnet_mode
        new_enabled = s["enabled"]
        new_mode = s["mode"]
        new_cached_attn = s.get("use_cached_attn")

        # Check if cached attention setting changed
        cached_attn_changed = False
        if new_cached_attn is not None and new_cached_attn != self.use_cached_attn:
            self.use_cached_attn = new_cached_attn
            cached_attn_changed = True

        if new_enabled == old_enabled and new_mode == old_mode and not cached_attn_changed:
            return False

        # Determine if the ControlNet model itself changes
        def _cn_model_key(mode):
            return "depth" if mode.startswith("depth_") else mode

        model_changed = (
            cached_attn_changed
            or new_enabled != old_enabled
            or _cn_model_key(new_mode) != _cn_model_key(old_mode)
        )

        self.controlnet_enabled = new_enabled
        self.controlnet_mode = new_mode

        # Load MiDaS depth estimator if switching to depth_midas
        if new_enabled and new_mode == "depth_midas" and self._depth_estimator is None:
            from depth_estimator import DepthEstimator
            self._depth_estimator = DepthEstimator()

        if not model_changed:
            return False
        was_enabled = self.enabled
        self.enabled = False
        self.status = "Reloading model..."
        self.stream = None
        self.init_model()
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt="blurry, low quality, distorted, artifacts, ugly",
            num_inference_steps=50,
            guidance_scale=2.0,
            delta=self.delta,
        )
        # Warmup
        dummy = Image.new("RGB", (320, 240))
        tensor = self.stream.preprocess_image(dummy)
        for _ in range(5):
            self.stream(image=tensor)
        torch.cuda.synchronize()
        self.enabled = was_enabled
        self.status = "Running" if self.enabled else "Ready (diffusion OFF)"
        return True

    def set_raw_frame(self, frame):
        self.latest_raw_frame = frame.copy()

    def process_loop(self):
        count = 0
        t0 = time.time()

        while self.running:
            self._apply_pending()
            if self._apply_controlnet_reload():
                continue

            if not self.enabled or self.latest_raw_frame is None:
                time.sleep(0.005)
                continue

            try:
                raw_frame = self.latest_raw_frame
                flow = None
                curr_z14 = None

                # Compute optical flow / motion vectors for temporal modes
                if self.temporal_mode in ("flow", "v2v+flow"):
                    curr_gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
                    if self._prev_raw_gray is not None:
                        if (self.motion_source == "game_state" and
                                self._state_reader is not None and
                                self._rdram is not None):
                            # Try game-state motion vectors first
                            if self._prev_z14 is not None:
                                flow = self._state_reader.compute_motion_vectors(
                                    self._prev_z14, curr_gray.shape[1], curr_gray.shape[0])
                            if flow is None:
                                flow = self._temporal_blender.compute_flow(
                                    self._prev_raw_gray, curr_gray)
                        else:
                            flow = self._temporal_blender.compute_flow(
                                self._prev_raw_gray, curr_gray)

                        # Flow debug visualization
                        if self.show_flow_debug and flow is not None:
                            self.latest_flow_debug = self._temporal_blender.visualize_flow(flow)

                    self._prev_raw_gray = curr_gray

                    # Z-buffer for occlusion detection
                    if self._rdram is not None:
                        curr_z14, _ = _decode_n64_z(
                            self._rdram, SM64_ZBUFFER_ADDR, 320, 240)

                # Warped noise (optional, requires flow)
                if self.use_warped_noise and flow is not None:
                    self.stream.stream.set_noise_flow(flow)
                else:
                    self.stream.stream.set_noise_flow(None)

                # Compute ControlNet conditioning
                if self.controlnet_enabled and self.stream.stream.controlnet is not None:
                    if self.controlnet_mode == "canny":
                        cond_image = extract_canny(raw_frame, self.canny_low, self.canny_high)
                    elif self.controlnet_mode == "depth_zbuffer" and self._rdram is not None:
                        if curr_z14 is not None:
                            # Reuse z14 we already decoded
                            z_norm = curr_z14 - curr_z14.min()
                            z_range = curr_z14.max() - curr_z14.min()
                            if z_range > 0:
                                z_norm = z_norm / z_range
                            z_u8 = (z_norm * 255).astype(np.uint8)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            cond_image = cv2.cvtColor(clahe.apply(z_u8), cv2.COLOR_GRAY2RGB)
                        else:
                            cond_image = read_n64_zbuffer(self._rdram)
                        if self.show_zbuffer_debug:
                            self.latest_zbuffer_debug = cond_image
                    else:  # depth_midas
                        cond_image = self._depth_estimator.estimate(raw_frame)
                    cond_tensor = self.stream.preprocess_controlnet_image(cond_image)
                    self.stream.set_controlnet_cond(cond_tensor)

                # Run diffusion
                pil_img = Image.fromarray(raw_frame)
                tensor = self.stream.preprocess_image(pil_img)
                result = self.stream(image=tensor)
                if isinstance(result, list):
                    result = result[0]
                new_frame = np.array(result)

                # Temporal blending
                if self.temporal_mode in ("flow", "v2v+flow") and flow is not None:
                    # Flow-guided blend with occlusion
                    if self._prev_enhanced is not None and self._prev_enhanced.shape == new_frame.shape:
                        # Compute occlusion mask at raw flow resolution (matches z-buffer size)
                        if (curr_z14 is not None and self._prev_z14 is not None):
                            occ_mask = self._temporal_blender.compute_occlusion_mask(
                                self._prev_z14, curr_z14, flow,
                                self.occlusion_threshold)
                        else:
                            occ_mask = np.zeros(flow.shape[:2], dtype=np.float32)

                        # Rescale flow and occlusion mask to match enhanced frame resolution
                        fh, fw = new_frame.shape[:2]
                        if flow.shape[:2] != (fh, fw):
                            sy = fh / flow.shape[0]
                            sx = fw / flow.shape[1]
                            flow = cv2.resize(flow, (fw, fh))
                            flow[:, :, 0] *= sx
                            flow[:, :, 1] *= sy
                            occ_mask = cv2.resize(occ_mask, (fw, fh))

                        new_frame = self._temporal_blender.blend(
                            new_frame, self._prev_enhanced, flow, occ_mask,
                            self.flow_blend_strength)
                    self._prev_z14 = curr_z14
                elif self.temporal_mode == "naive":
                    # Legacy simple alpha blend
                    alpha = self.temporal_blend
                    if alpha > 0 and self._prev_enhanced is not None and self._prev_enhanced.shape == new_frame.shape:
                        new_frame = cv2.addWeighted(
                            new_frame, 1.0 - alpha, self._prev_enhanced, alpha, 0)
                # "off" and "v2v" (feature bank only) — no pixel-space blending

                self.latest_enhanced_frame = new_frame
                self._prev_enhanced = new_frame

                count += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.diff_fps = count / (now - t0)
                    count = 0
                    t0 = now
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status = f"Error: {e}"
                time.sleep(0.5)


# ============================================================
# tkinter Control Panel
# ============================================================

def create_control_panel(processor):
    root = tk.Tk()
    root.title("N64 DLSS Control Panel")
    root.geometry("620x720")
    root.resizable(True, True)

    enabled_var = tk.BooleanVar(value=False)
    prompt_var = tk.StringVar(value=processor.prompt)
    strength_var = tk.DoubleVar(value=processor.strength)
    delta_var = tk.DoubleVar(value=processor.delta)
    status_var = tk.StringVar(value="Ready")
    diff_fps_var = tk.StringVar(value="Diffusion: -- FPS")
    emu_fps_var = tk.StringVar(value="Emulator: -- FPS")

    # ---- Two-column layout ----
    columns = tk.Frame(root)
    columns.pack(fill="both", expand=True, padx=4, pady=4)

    left_col = tk.Frame(columns)
    left_col.pack(side="left", fill="both", expand=True, padx=(0, 2))

    right_col = tk.Frame(columns)
    right_col.pack(side="left", fill="both", expand=True, padx=(2, 0))

    # ==========================================================
    # LEFT COLUMN
    # ==========================================================

    # -- Control --
    f_ctrl = ttk.LabelFrame(left_col, text="Control", padding=8)
    f_ctrl.pack(fill="x", pady=4)

    def on_toggle():
        processor.enabled = enabled_var.get()
        if processor.enabled:
            processor.status = "Running"
        else:
            processor.status = "Ready (diffusion OFF)"

    w = ttk.Checkbutton(f_ctrl, text="Enable Diffusion  [F2]",
                        variable=enabled_var, command=on_toggle)
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["enable_diffusion"])

    # -- Settings --
    f_set = ttk.LabelFrame(left_col, text="Settings", padding=8)
    f_set.pack(fill="x", pady=4)

    ttk.Label(f_set, text="Prompt:").pack(anchor="w")
    prompt_entry = ttk.Entry(f_set, textvariable=prompt_var, width=32)
    prompt_entry.pack(fill="x", pady=(0, 6))
    ToolTip(prompt_entry, TOOLTIPS["prompt"])

    strength_label = ttk.Label(f_set, text=f"Denoise Strength: {strength_var.get():.2f}")
    strength_label.pack(anchor="w")

    def on_strength_slide(val):
        strength_label.config(text=f"Denoise Strength: {float(val):.2f}")

    w = ttk.Scale(f_set, from_=0.0, to=1.0, variable=strength_var,
                  orient="horizontal", command=on_strength_slide)
    w.pack(fill="x")
    ToolTip(w, TOOLTIPS["denoise_strength"])

    hint = tk.Frame(f_set)
    hint.pack(fill="x")
    ttk.Label(hint, text="<- Faithful", font=("", 7)).pack(side="left")
    ttk.Label(hint, text="Creative ->", font=("", 7)).pack(side="right")

    tk.Frame(f_set, height=6).pack()

    d_label = ttk.Label(f_set, text=f"Delta: {delta_var.get():.2f}")
    d_label.pack(anchor="w")

    def on_d_slide(val):
        d_label.config(text=f"Delta: {float(val):.2f}")

    w = ttk.Scale(f_set, from_=0.0, to=1.0, variable=delta_var,
                  orient="horizontal", command=on_d_slide)
    w.pack(fill="x")
    ToolTip(w, TOOLTIPS["delta"])

    tk.Frame(f_set, height=6).pack()

    blend_var = tk.DoubleVar(value=processor.temporal_blend)
    blend_label = ttk.Label(f_set, text=f"Temporal Blend: {blend_var.get():.2f}")
    blend_label.pack(anchor="w")

    def on_blend_slide(val):
        v = float(val)
        blend_label.config(text=f"Temporal Blend: {v:.2f}")
        processor.temporal_blend = v

    blend_scale = ttk.Scale(f_set, from_=0.0, to=0.9, variable=blend_var,
                            orient="horizontal", command=on_blend_slide)
    blend_scale.pack(fill="x")
    ToolTip(blend_scale, TOOLTIPS["temporal_blend"])

    blend_hint = tk.Frame(f_set)
    blend_hint.pack(fill="x")
    ttk.Label(blend_hint, text="<- Responsive", font=("", 7)).pack(side="left")
    ttk.Label(blend_hint, text="Stable ->", font=("", 7)).pack(side="right")

    # Group blend widgets for conditional activation
    blend_widgets = [blend_label, blend_scale, blend_hint]

    def apply_settings():
        processor.request_settings(
            prompt=prompt_var.get(),
            strength=float(strength_var.get()),
            delta=float(delta_var.get()),
        )

    w = ttk.Button(f_set, text="Apply Settings", command=apply_settings)
    w.pack(fill="x", pady=(8, 0))
    ToolTip(w, TOOLTIPS["apply_settings"])
    prompt_entry.bind("<Return>", lambda e: apply_settings())

    # -- Presets --
    f_pre = ttk.LabelFrame(left_col, text="Presets", padding=8)
    f_pre.pack(fill="x", pady=4)

    def apply_preset(name):
        p = PRESETS[name]
        prompt_var.set(p["prompt"])
        strength_var.set(p["strength"])
        delta_var.set(p["delta"])
        on_strength_slide(p["strength"])
        on_d_slide(p["delta"])
        apply_settings()

    for name in PRESETS:
        ttk.Button(f_pre, text=name,
                   command=lambda n=name: apply_preset(n)).pack(fill="x", pady=1)

    # -- Status --
    f_stat = ttk.LabelFrame(left_col, text="Status", padding=8)
    f_stat.pack(fill="x", pady=4)

    ttk.Label(f_stat, textvariable=diff_fps_var).pack(anchor="w")
    ttk.Label(f_stat, textvariable=emu_fps_var).pack(anchor="w")
    ttk.Label(f_stat, textvariable=status_var).pack(anchor="w")

    # ==========================================================
    # RIGHT COLUMN
    # ==========================================================

    # -- ControlNet --
    f_cn = ttk.LabelFrame(right_col, text="ControlNet", padding=8)
    f_cn.pack(fill="x", pady=4)

    cn_enabled_var = tk.BooleanVar(value=processor.controlnet_enabled)
    cn_mode_var = tk.StringVar(value=processor.controlnet_mode)
    cn_scale_var = tk.DoubleVar(value=processor.controlnet_scale)
    cn_canny_low_var = tk.IntVar(value=processor.canny_low)
    cn_canny_high_var = tk.IntVar(value=processor.canny_high)

    w = ttk.Checkbutton(f_cn, text="Enable ControlNet (requires reload)",
                        variable=cn_enabled_var)
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["cn_enable"])

    mode_frame = tk.Frame(f_cn)
    mode_frame.pack(fill="x", pady=2)
    w = ttk.Radiobutton(mode_frame, text="Canny", variable=cn_mode_var,
                        value="canny")
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["cn_canny"])
    w = ttk.Radiobutton(mode_frame, text="Depth (MiDaS)", variable=cn_mode_var,
                        value="depth_midas")
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["cn_depth_midas"])
    w = ttk.Radiobutton(mode_frame, text="Depth (N64 Z-Buffer)", variable=cn_mode_var,
                        value="depth_zbuffer")
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["cn_depth_zbuffer"])

    cn_scale_label = ttk.Label(f_cn, text=f"Scale: {cn_scale_var.get():.2f}")
    cn_scale_label.pack(anchor="w")

    def on_cn_scale(val):
        cn_scale_label.config(text=f"Scale: {float(val):.2f}")
        processor.request_controlnet_settings(scale=float(val))

    cn_scale_slider = ttk.Scale(f_cn, from_=0.0, to=2.0, variable=cn_scale_var,
                                orient="horizontal", command=on_cn_scale)
    cn_scale_slider.pack(fill="x")
    ToolTip(cn_scale_slider, TOOLTIPS["cn_scale"])

    # Group scale widgets for conditional activation
    cn_scale_widgets = [cn_scale_label, cn_scale_slider]

    canny_frame = ttk.LabelFrame(f_cn, text="Canny Thresholds", padding=4)
    canny_frame.pack(fill="x", pady=2)

    canny_low_label = ttk.Label(canny_frame, text=f"Low: {cn_canny_low_var.get()}")
    canny_low_label.pack(anchor="w")

    def on_canny_low(val):
        v = int(float(val))
        canny_low_label.config(text=f"Low: {v}")
        processor.request_controlnet_settings(canny_low=v)

    canny_low_scale = ttk.Scale(canny_frame, from_=0, to=255, variable=cn_canny_low_var,
                                orient="horizontal", command=on_canny_low)
    canny_low_scale.pack(fill="x")
    ToolTip(canny_low_scale, TOOLTIPS["cn_canny_low"])

    canny_high_label = ttk.Label(canny_frame, text=f"High: {cn_canny_high_var.get()}")
    canny_high_label.pack(anchor="w")

    def on_canny_high(val):
        v = int(float(val))
        canny_high_label.config(text=f"High: {v}")
        processor.request_controlnet_settings(canny_high=v)

    canny_high_scale = ttk.Scale(canny_frame, from_=0, to=255, variable=cn_canny_high_var,
                                 orient="horizontal", command=on_canny_high)
    canny_high_scale.pack(fill="x")
    ToolTip(canny_high_scale, TOOLTIPS["cn_canny_high"])

    zbuf_debug_var = tk.BooleanVar(value=False)

    def on_zbuf_debug():
        processor.show_zbuffer_debug = zbuf_debug_var.get()
        if not zbuf_debug_var.get():
            processor.latest_zbuffer_debug = None

    w = ttk.Checkbutton(f_cn, text="Show Z-Buffer Debug",
                        variable=zbuf_debug_var, command=on_zbuf_debug)
    w.pack(anchor="w", pady=(4, 0))
    ToolTip(w, TOOLTIPS["cn_zbuf_debug"])

    def save_zbuffer_snapshot():
        if processor._rdram is not None:
            dump_zbuffer_snapshot(processor._rdram, processor.latest_raw_frame)
        else:
            print("No RDRAM available for Z-buffer snapshot")

    w = ttk.Button(f_cn, text="Save Z-Buffer Snapshot",
                   command=save_zbuffer_snapshot)
    w.pack(fill="x", pady=(4, 0))
    ToolTip(w, TOOLTIPS["cn_zbuf_snapshot"])

    def reload_controlnet():
        processor.request_reload(
            controlnet_enabled=cn_enabled_var.get(),
            controlnet_mode=cn_mode_var.get(),
        )

    w = ttk.Button(f_cn, text="Reload Model", command=reload_controlnet)
    w.pack(fill="x", pady=(6, 0))
    ToolTip(w, TOOLTIPS["cn_reload"])

    # -- Temporal Consistency --
    f_tc = ttk.LabelFrame(right_col, text="Temporal Consistency", padding=8)
    f_tc.pack(fill="x", pady=4)

    tc_mode_var = tk.StringVar(value=processor.temporal_mode)
    tc_modes = [("Off", "off"), ("Naive Blend", "naive"), ("Flow-Guided", "flow"),
                ("Feature Bank (V2V)", "v2v"), ("V2V + Flow", "v2v+flow")]

    ttk.Label(f_tc, text="Temporal Mode:").pack(anchor="w")
    tc_combo = ttk.Combobox(f_tc, textvariable=tc_mode_var, state="readonly",
                            values=[m[0] for m in tc_modes], width=20)
    for i, (label, val) in enumerate(tc_modes):
        if val == processor.temporal_mode:
            tc_combo.current(i)
            break
    tc_combo.pack(fill="x", pady=(0, 4))
    ToolTip(tc_combo, TOOLTIPS["tc_mode"])

    flow_strength_var = tk.DoubleVar(value=processor.flow_blend_strength)
    flow_strength_label = ttk.Label(f_tc, text=f"Flow Blend: {flow_strength_var.get():.2f}")
    flow_strength_label.pack(anchor="w")

    def on_flow_strength(val):
        v = float(val)
        flow_strength_label.config(text=f"Flow Blend: {v:.2f}")
        processor.flow_blend_strength = v

    flow_strength_scale = ttk.Scale(f_tc, from_=0.0, to=0.9, variable=flow_strength_var,
                                    orient="horizontal", command=on_flow_strength)
    flow_strength_scale.pack(fill="x")
    ToolTip(flow_strength_scale, TOOLTIPS["tc_flow_blend"])

    occ_thresh_var = tk.DoubleVar(value=processor.occlusion_threshold)
    occ_label = ttk.Label(f_tc, text=f"Occlusion Thresh: {occ_thresh_var.get():.2f}")
    occ_label.pack(anchor="w")

    def on_occ_thresh(val):
        v = float(val)
        occ_label.config(text=f"Occlusion Thresh: {v:.2f}")
        processor.occlusion_threshold = v

    occ_scale = ttk.Scale(f_tc, from_=0.01, to=0.5, variable=occ_thresh_var,
                          orient="horizontal", command=on_occ_thresh)
    occ_scale.pack(fill="x")
    ToolTip(occ_scale, TOOLTIPS["tc_occ_thresh"])

    fi_strength_var = tk.DoubleVar(value=processor.feature_injection_strength)
    fi_label = ttk.Label(f_tc, text=f"Feature Injection: {fi_strength_var.get():.2f}")
    fi_label.pack(anchor="w")

    def on_fi_strength(val):
        v = float(val)
        fi_label.config(text=f"Feature Injection: {v:.2f}")
        processor.feature_injection_strength = v
        # Update live if model supports it
        if (processor.stream and processor.use_cached_attn):
            try:
                from streamdiffusion.models.cache_utils import update_feature_injection_strength
                update_feature_injection_strength(processor.stream.stream.unet, v)
            except Exception:
                pass

    fi_scale = ttk.Scale(f_tc, from_=0.0, to=1.0, variable=fi_strength_var,
                         orient="horizontal", command=on_fi_strength)
    fi_scale.pack(fill="x")
    ToolTip(fi_scale, TOOLTIPS["tc_fi_strength"])

    fi_thresh_var = tk.DoubleVar(value=processor.feature_similarity_threshold)
    fi_thresh_label = ttk.Label(f_tc, text=f"Similarity Thresh: {fi_thresh_var.get():.2f}")
    fi_thresh_label.pack(anchor="w")

    def on_fi_thresh(val):
        v = float(val)
        fi_thresh_label.config(text=f"Similarity Thresh: {v:.2f}")
        processor.feature_similarity_threshold = v
        if (processor.stream and processor.use_cached_attn):
            try:
                from streamdiffusion.models.cache_utils import update_feature_similarity_threshold
                update_feature_similarity_threshold(processor.stream.stream.unet, v)
            except Exception:
                pass

    fi_thresh_scale = ttk.Scale(f_tc, from_=0.90, to=0.99, variable=fi_thresh_var,
                                orient="horizontal", command=on_fi_thresh)
    fi_thresh_scale.pack(fill="x")
    ToolTip(fi_thresh_scale, TOOLTIPS["tc_fi_thresh"])

    warped_noise_var = tk.BooleanVar(value=processor.use_warped_noise)

    def on_warped_noise():
        processor.use_warped_noise = warped_noise_var.get()

    warped_noise_cb = ttk.Checkbutton(f_tc, text="Warped Noise", variable=warped_noise_var,
                                      command=on_warped_noise)
    warped_noise_cb.pack(anchor="w", pady=(4, 0))
    ToolTip(warped_noise_cb, TOOLTIPS["tc_warped_noise"])

    # Motion source radio buttons
    motion_src_var = tk.StringVar(value=processor.motion_source)
    motion_frame = tk.Frame(f_tc)
    motion_frame.pack(fill="x", pady=2)
    ttk.Label(motion_frame, text="Motion Source:").pack(anchor="w")

    def on_motion_src():
        processor.motion_source = motion_src_var.get()

    w = ttk.Radiobutton(motion_frame, text="Optical Flow", variable=motion_src_var,
                        value="optical_flow", command=on_motion_src)
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["tc_motion_optical"])
    w = ttk.Radiobutton(motion_frame, text="Game State", variable=motion_src_var,
                        value="game_state", command=on_motion_src)
    w.pack(anchor="w")
    ToolTip(w, TOOLTIPS["tc_motion_gamestate"])

    flow_debug_var = tk.BooleanVar(value=False)

    def on_flow_debug():
        processor.show_flow_debug = flow_debug_var.get()
        if not flow_debug_var.get():
            processor.latest_flow_debug = None

    flow_debug_cb = ttk.Checkbutton(f_tc, text="Show Flow Debug", variable=flow_debug_var,
                                    command=on_flow_debug)
    flow_debug_cb.pack(anchor="w", pady=(4, 0))
    ToolTip(flow_debug_cb, TOOLTIPS["tc_flow_debug"])

    def apply_temporal_mode(event=None):
        selected_label = tc_combo.get()
        for label, val in tc_modes:
            if label == selected_label:
                old_mode = processor.temporal_mode
                needs_v2v = val in ("v2v", "v2v+flow")
                had_v2v = old_mode in ("v2v", "v2v+flow")

                processor.temporal_mode = val

                # V2V mode change requires model reload (attention processors change)
                if needs_v2v != had_v2v:
                    processor.request_reload(
                        controlnet_enabled=cn_enabled_var.get(),
                        controlnet_mode=cn_mode_var.get(),
                        use_cached_attn=needs_v2v,
                    )
                # Reset temporal state when switching modes
                processor._prev_raw_gray = None
                processor._prev_z14 = None
                processor._prev_enhanced = None
                break
        _update_widget_states()

    tc_combo.bind("<<ComboboxSelected>>", apply_temporal_mode)

    # ==========================================================
    # Conditional activation logic
    # ==========================================================

    def _update_widget_states():
        mode = processor.temporal_mode
        cn_on = cn_enabled_var.get()
        cn_mode = cn_mode_var.get()

        # Temporal Blend — only for naive mode
        is_naive = (mode == "naive")
        for wgt in blend_widgets:
            set_widget_state(wgt, is_naive)

        # ControlNet Scale — only when CN enabled
        for wgt in cn_scale_widgets:
            set_widget_state(wgt, cn_on)

        # Canny Thresholds — CN enabled AND canny mode
        set_widget_state(canny_frame, cn_on and cn_mode == "canny")

        # Flow-related controls
        has_flow = mode in ("flow", "v2v+flow")
        set_widget_state(flow_strength_label, has_flow)
        set_widget_state(flow_strength_scale, has_flow)
        set_widget_state(occ_label, has_flow)
        set_widget_state(occ_scale, has_flow)
        set_widget_state(warped_noise_cb, has_flow)
        set_widget_state(motion_frame, has_flow)
        set_widget_state(flow_debug_cb, has_flow)

        # Feature injection controls — v2v or v2v+flow
        has_v2v = mode in ("v2v", "v2v+flow")
        set_widget_state(fi_label, has_v2v)
        set_widget_state(fi_scale, has_v2v)
        set_widget_state(fi_thresh_label, has_v2v)
        set_widget_state(fi_thresh_scale, has_v2v)

    # Trigger on ControlNet variable changes
    cn_enabled_var.trace_add("write", lambda *_: _update_widget_states())
    cn_mode_var.trace_add("write", lambda *_: _update_widget_states())

    # Set initial state
    _update_widget_states()

    root._v = {
        "enabled": enabled_var,
        "status": status_var,
        "diff_fps": diff_fps_var,
        "emu_fps": emu_fps_var,
    }
    return root


# ============================================================
# Main
# ============================================================

def main():
    import pygame
    from n64_frontend import N64Frontend

    rom_path = os.path.join(SCRIPT_DIR, "n64emulator", "sm65.z64")
    core_path = os.path.join(SCRIPT_DIR, "n64emulator", "cores",
                             "parallel_n64_libretro.dll")

    if not os.path.exists(rom_path):
        print(f"ROM not found: {rom_path}")
        sys.exit(1)
    if not os.path.exists(core_path):
        print(f"Core not found: {core_path}")
        sys.exit(1)

    # ---- Init diffusion (before pygame/OpenGL to avoid context conflicts) ----
    processor = DiffusionProcessor()
    processor.init_model()

    diff_thread = threading.Thread(target=processor.process_loop, daemon=True)
    diff_thread.start()

    # ---- Init emulator (creates pygame+OpenGL window) ----
    frontend = N64Frontend(core_path)
    frontend.on_frame = processor.set_raw_frame
    frontend.init()
    frontend.load_game(rom_path)

    # Wire RDRAM for Z-buffer depth
    rdram, rdram_size = frontend.get_rdram()
    if rdram:
        processor.set_rdram(rdram)
        print(f"RDRAM access: {rdram_size / 1024 / 1024:.1f} MB")
    else:
        print("WARNING: Could not access RDRAM — Z-buffer depth unavailable")

    # ---- Init tkinter control panel ----
    root = create_control_panel(processor)

    # ---- Raw frame comparison window ----
    from PIL import ImageTk
    raw_win = tk.Toplevel(root)
    raw_win.title("N64 Raw (No AI)")
    raw_win.geometry("640x480")
    raw_win.resizable(True, True)
    raw_canvas = tk.Label(raw_win)
    raw_canvas.pack(fill="both", expand=True)
    raw_win._photo = None  # prevent GC

    # ---- Grab the frontend's OpenGL display setup ----
    from OpenGL import GL
    frontend._init_gl_display()

    screen = frontend._screen
    disp_w, disp_h = screen.get_size()
    clock = pygame.time.Clock()

    running = True
    emu_count = 0
    emu_t0 = time.time()
    emu_fps = 0.0

    print("\nRunning! F2 = Toggle diffusion | ESC = Quit")

    while running:
        # -- tkinter pump --
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break

        # -- Update status --
        root._v["status"].set(processor.status)
        root._v["diff_fps"].set(f"Diffusion: {processor.diff_fps:.1f} FPS")
        root._v["emu_fps"].set(f"Emulator: {emu_fps:.1f} FPS")

        # -- Pygame events --
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    frontend.core.retro_reset()
                elif event.key == pygame.K_F2:
                    processor.enabled = not processor.enabled
                    root._v["enabled"].set(processor.enabled)
                    if processor.enabled:
                        processor.status = "Running"
                    else:
                        processor.status = "Ready (diffusion OFF)"
                else:
                    frontend.keys_pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                frontend.keys_pressed.discard(event.key)
            elif event.type == pygame.VIDEORESIZE:
                disp_w, disp_h = event.w, event.h

        # -- Emulator step --
        frontend.core.retro_run()

        # -- Update raw frame comparison window --
        if processor.latest_raw_frame is not None:
            try:
                raw_img = Image.fromarray(processor.latest_raw_frame)
                w, h = raw_canvas.winfo_width(), raw_canvas.winfo_height()
                if w > 1 and h > 1:
                    raw_img = raw_img.resize((w, h), Image.NEAREST)
                raw_win._photo = ImageTk.PhotoImage(raw_img)
                raw_canvas.configure(image=raw_win._photo)
            except tk.TclError:
                pass  # window closed

        # -- Swap in enhanced frame if diffusion is on --
        if processor.enabled and processor.latest_enhanced_frame is not None:
            frontend.frame_data = processor.latest_enhanced_frame

        # -- Update Z-buffer debug (runs independently of ControlNet) --
        if processor.show_zbuffer_debug and processor._rdram is not None:
            processor.latest_zbuffer_debug = read_n64_zbuffer(processor._rdram)

        # -- Draw using frontend's GL renderer --
        if frontend.hw_render:
            pass  # core already rendered to GL
        else:
            frontend._draw_frame_gl(disp_w, disp_h)

        # -- Z-Buffer debug overlay --
        if processor.show_zbuffer_debug and processor.latest_zbuffer_debug is not None:
            zbuf = processor.latest_zbuffer_debug
            # Quarter-size overlay in bottom-right corner
            oh, ow = zbuf.shape[:2]
            oh, ow = oh // 2, ow // 2
            small = cv2.resize(zbuf, (ow, oh))
            # Flip vertically — OpenGL draws bottom-up, our image is top-down
            small = small[::-1].copy()
            # Brighten for visibility in the overlay
            small = cv2.convertScaleAbs(small, alpha=1.8, beta=30)
            GL.glWindowPos2i(disp_w - ow, 0)
            GL.glDrawPixels(ow, oh, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, small.tobytes())

        # -- Flow debug overlay --
        if processor.show_flow_debug and processor.latest_flow_debug is not None:
            flow_vis = processor.latest_flow_debug
            oh, ow = flow_vis.shape[:2]
            oh, ow = oh // 2, ow // 2
            small = cv2.resize(flow_vis, (ow, oh))
            small = small[::-1].copy()
            GL.glWindowPos2i(disp_w - ow, oh)  # Above Z-buffer debug
            GL.glDrawPixels(ow, oh, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, small.tobytes())

        pygame.display.flip()

        # -- FPS --
        emu_count += 1
        now = time.time()
        if now - emu_t0 >= 1.0:
            emu_fps = emu_count / (now - emu_t0)
            emu_count = 0
            emu_t0 = now
            pygame.display.set_caption(
                f"N64 DLSS Live - Emu {emu_fps:.0f} FPS | "
                f"Diff {'ON' if processor.enabled else 'OFF'} "
                f"({processor.diff_fps:.0f} FPS)"
            )

        clock.tick(frontend.target_fps)

    # -- Cleanup --
    processor.running = False
    diff_thread.join(timeout=2)
    if frontend.hw_render and frontend.hw_context_destroy:
        frontend.hw_context_destroy()
    frontend.core.retro_unload_game()
    frontend.core.retro_deinit()
    pygame.quit()
    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
