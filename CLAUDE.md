# N64 Frame Gen — Project Guide

## Overview
Real-time neural frame enhancement for N64 emulation. Captures frames from a libretro N64 core, enhances them with AI, displays in real time.

## Current State (v1 — StreamDiffusion)
- Uses SD-Turbo via StreamDiffusion + TensorRT at ~39 FPS on GTX 4060
- Two-column tkinter control panel with tooltips and conditional activation
- ControlNet support (canny, MiDaS depth, N64 Z-buffer depth)
- Temporal consistency (naive blend, flow-guided, V2V feature bank)
- Raw frame comparison window
- Source code in `src/`, launched via `run.bat` or `python src/n64_dlss_live.py`
- Python 3.10 venv at `venv310/`

## Next Phase: G-Buffer Conditioned Style Transfer (v2)

### Plan
Train a LoRA fine-tuned diffusion model that accepts depth + normals as native input channels alongside the latent image, enabling prompt-controllable style transfer with perfect structural preservation.

### Steps
1. **Compute normals from Z-buffer** — cross-product of depth gradients (30 min)
2. **Build data capture mode** — auto-save RGB + depth + normals during gameplay (1-2 hrs code + days of gameplay, target 50K frames)
3. **Generate styled training targets** — SDXL + ControlNet-Depth on cloud A100 (~$28/style, ~14 hrs)
4. **Prepare training pipeline** — diffusers + peft, expand conv_in 4→8 channels, LoRA rank 32
5. **Train on cloud** — A100/H100, 8-16 hours, ~$16-32
6. **Export + optimize** — TensorRT conversion, integrate into emulator
7. **Test + iterate**

### Budget
~$100 allocated. Use cloud (Vast.ai / Lambda / RunPod) with powerful GPUs for both target generation and training.

### Base Model Decision
SD-Turbo (SD 2.1 distilled) is the current base. Consider SD 1.5 + LCM-LoRA as alternative — much larger LoRA/ControlNet ecosystem, community style LoRAs work out of the box, 2-4 step inference. SDXL-Turbo has better quality but tight on 8GB consumer GPUs.

### Key Insight
Expanding input channels to include depth + normals eliminates the "dead zone" problem (style vs structure tradeoff). The model can apply strong styles while perfectly preserving geometry because it has explicit 3D structure data.

## Architecture
- `src/n64_dlss_live.py` — main app, DiffusionProcessor, control panel, main loop
- `src/ui_utils.py` — ToolTip, set_widget_state, TOOLTIPS dict
- `src/temporal_blend.py` — optical flow warping, occlusion detection, TAA clamping
- `src/noise_warping.py` — flow-warped noise for coherent denoising
- `src/sm64_state_reader.py` — SM64 game state from RDRAM
- `src/depth_estimator.py` — MiDaS depth estimation
- `n64Emulator/n64_frontend.py` — libretro frontend (ctypes + pygame + OpenGL)

## Dev Notes
- V2V modes (Feature Bank) require PyTorch fallback — TensorRT UNet has no `attn_processors`
- N64 core outputs 640x240 frames, not 320x240 — flow/occlusion must handle resolution mismatches
- Z-buffer unpacking: `curr_z14, _ = _decode_n64_z(...)` (first return value is the 2D depth, second is raw z16)
- `processor.temporal_mode` drives conditional widget activation in the control panel
