# N64 DLSS - ControlNet Integration Scratchpad

## The Problem
SD-Turbo img2img with StreamDiffusion has a dead zone:
- Low t_index (0-30): loses the original scene entirely
- High t_index (40-45): preserves scene but prompt has no visible effect
- No sweet spot for "same scene, different style"

## The Solution: ControlNet / T2I-Adapter
Extract structural signals (depth, edges) from the N64 emulator and use them
to LOCK the scene layout while letting the model restyle freely.

## Why N64 Emulators Are Perfect For This
Emulators have access to the full rendering pipeline, not just the final frame:
- **Depth buffer (z-buffer)** — N64 uses hardware z-buffering, basically free to extract
- **Surface normals** — derivable from geometry
- **Edge/contour maps** — from depth discontinuities or Canny on the frame
- **Raw polygon/wireframe data** — the actual 3D scene structure

A perfect depth map from the emulator > estimated depth from monocular depth models.

## ControlNet Options
| Option | Extra Compute | Quality | Ease |
|--------|--------------|---------|------|
| ControlNet (depth) | ~60-80% of UNet | Best structural control | Moderate |
| T2I-Adapter (depth) | ~10-15% of UNet | Good, lighter conditioning | Easier |
| IP-Adapter | ~15-20% | Image embedding, less precise | Easiest |

## Performance Estimates (on GTX 4060, 320x240)
- Current (no ControlNet, TRT): 25.7ms / 39 FPS
- With ControlNet: ~35-40ms / 25-28 FPS (still above 24 FPS target)
- With T2I-Adapter: ~28-30ms / 33-36 FPS (minimal hit)

## Implementation Plan
1. Keep StreamDiffusion for real-time pipeline optimizations
2. Modify StreamDiffusion's denoising loop to inject ControlNet residuals:
   - Load ControlNet model alongside UNet
   - In unet_step: run control image through ControlNet encoder
   - Feed residuals into main UNet forward pass
3. Build TRT engines for ControlNet too (same ONNX export + engine build flow)

## Speed Tricks
- Cache ControlNet output when depth map hasn't changed (similarity filter)
- Run ControlNet at lower resolution than main UNet
- T2I-Adapter as lighter alternative (~77% fewer params than ControlNet)
- Skip ControlNet on frames where camera hasn't moved (stochastic filter)

## Emulators That May Expose Depth Buffer
- **Mupen64Plus** — open source, plugin architecture, video plugins could expose z-buffer
- **simple64** — modern fork of Mupen64Plus
- **Project64** — popular but less open
- **parallel-rdp** (Vulkan RDP plugin) — directly accesses RDP state including z-buffer

## Key Diffusers Models to Use
- `lllyasviel/control_v11f1p_sd15_depth` — SD 1.5 depth ControlNet
- `TencentARC/t2i-adapter-depth-midas-sdxl-1.0` — T2I-Adapter for depth
- SD-Turbo is SD 2.1 based, need compatible ControlNet (may need community model)
