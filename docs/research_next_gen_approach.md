# Next-Gen N64 Frame Enhancement: Research & Roadmap

## The Problem With Our Current Approach

Our current pipeline (StreamDiffusion + SD-Turbo + TensorRT) treats the N64 frame as a generic photograph and applies img2img diffusion. This has two fundamental issues:

1. **VRAM ceiling**: SD-Turbo needs ~4GB VRAM, leaving little room on an 8GB GPU. We're stuck with the smallest diffusion models.
2. **Information gap**: Diffusion models only see the final RGB pixels. Real DLSS gets depth, motion vectors, normals, material IDs, and lighting data directly from the rendering engine — that's why it's so much better.

## What Data Can We Actually Get From the N64 Emulator?

The N64's rendering pipeline (RSP → RDP → framebuffer) exposes far more than just pixels. Here's what's available, ordered by extraction difficulty:

### Already Have
| Data | Source | Notes |
|------|--------|-------|
| Color framebuffer | RDRAM read | Our raw frame |
| Z-buffer (14-bit depth) | RDRAM read | Already extracting |
| Motion vectors | Optical flow / SM64 game state | Already computing |

### Easy to Add (No Emulator Mods)
| Data | Source | How |
|------|--------|-----|
| **Screen-space normals** | Z-buffer gradients | `normal = normalize(cross(vec3(2,0,dz/dx), vec3(0,2,dz/dy)))` — 3 lines of code |
| **Surface slope (dZ)** | Z-buffer lower 2 bits | Already in the z16 data we read — `\|dZ/dx\| + \|dZ/dy\|` per pixel |
| **Fog color** | RDRAM constant | SetFogColor RDP command |

### Medium Effort (RDP Command Interception)
| Data | Source | How |
|------|--------|-----|
| **Color combiner mode** | RDP commands in RDRAM | Parse SetCombineMode — tells you material type (textured, vertex-colored, env-mapped, etc.) |
| **Lighting state** | Display list commands | Up to 7 directional lights + ambient, each with RGB color + direction |
| **Primitive/environment colors** | RDP commands | Constant colors used in the color combiner equation |
| **Texture data** | TMEM / RDRAM | Existing texture-dump infrastructure in Rice/GLideN64 |

### Hard (Requires HLE Interception or Emulator Mods)
| Data | Source | How |
|------|--------|-----|
| **Vertex normals** | Display list vertex data (bytes 12-14 when G_LIGHTING enabled) | Intercept GBI commands at RSP level |
| **Per-object IDs** | Display list hierarchy | Track G_DL call/return to assign IDs per object |
| **Albedo (texture-only)** | Color combiner decomposition | Separate TEXEL0 from SHADE in the combiner equation |
| **Transformation matrices** | G_MTX commands | Model-view and projection — gives camera and per-object transforms |

### Key Insight: Normals Are "Free"

The N64 vertex format stores normals in bytes 12-14 (as signed 8-bit nx/ny/nz) when `G_LIGHTING` is enabled. But even without intercepting vertices, we can compute **screen-space normals from the depth buffer** we already extract — this is standard in deferred rendering and requires only a cross-product of depth gradients.

---

## Three Paths Forward

### Path A: Drop-In Lightweight Super-Resolution (Quick Win)

**Replace diffusion entirely with a dedicated game upscaler.**

Models like **Real-ESRGAN** (animevideov3 variant, 1.3M params, 8MB) or **GameSR** achieve 60-240+ FPS using <500MB VRAM. That's 10x faster and 10x less VRAM than our current SD-Turbo pipeline.

**What it looks like:**
```
N64 frame (320x240) → Real-ESRGAN (TensorRT) → upscaled frame (1280x960)
```

**Pros:** Immediate, no training, huge performance gain, frees VRAM for other features.
**Cons:** Generic upscaling — doesn't use any emulator internals, limited to learned priors about "what sharp images look like."

**Key models:**
- `realesr-animevideov3` — 1.3M params, ~65 FPS at 640x480 on V100. Likely 100+ FPS at 320x240 on GTX 4060
- **GameSR** (OpenReview 2024) — purpose-built for game content, ConvLSTM temporal consistency, up to 240 FPS
- **LCS** (July 2025) — 0.21M params after reparameterization, beats AMD FSR1 on perceptual metrics

### Path B: Multi-Channel Conditioned Upscaler (Best Quality/Performance Tradeoff)

**Train a small CNN that takes RGB + depth + normals → enhanced frame.**

This is the approach used by DLSS 2.0 (before the transformer era) and all the recent academic papers. The key insight: a 2-5M parameter CNN with auxiliary inputs can match or exceed what a 860M parameter diffusion model does with RGB alone.

**What it looks like:**
```
N64 frame (320x240) ─┐
Z-buffer depth ───────┼→ Custom CNN (TensorRT) → enhanced frame (1280x960)
Computed normals ─────┤
Motion vectors ───────┘
```

**Architecture (based on NSRD / Meta Neural Supersampling / QRISP):**
- **Main branch:** Process RGB through a compact encoder-decoder
- **Auxiliary branch:** Lightweight feature extractor for depth + normals (these are clean but aliased — need less processing)
- **Temporal module:** ConvLSTM or recurrent feature warping using motion vectors to accumulate detail across frames
- **Output:** Predict upsampling kernels (per Arm's approach) or direct RGB + residual

**Training data generation:**
- Render each SM64 scene at 320x240 (input) and 1280x960 via paraLLEl-RDP upscaling (target)
- Extract depth, compute normals, compute optical flow — all automated
- Existing datasets: **GameIR** (19,200 paired frames from CARLA/UE4 with depth+segmentation), **SRGD** (14K frames from UE at 4 resolutions)
- Fine-tune on N64-specific pairs for best results

**VRAM:** <500MB for inference. Training fits in 8GB with gradient checkpointing.
**FPS:** 60-120 FPS with TensorRT (based on QRISP/MNSS benchmarks for similar architectures).

**Key papers:**
- *Neural Supersampling for Real-Time Rendering* (Meta, SIGGRAPH 2020) — foundational DLSS-like approach
- *NSRD: Neural SR with Radiance Demodulation* (CVPR 2024) — separate lighting from texture, upscale lighting smoothly
- *QRISP* (ICCV 2023) — 4x more efficient than prior neural supersampling
- *Efficient Video SR with Decoupled G-buffer Guidance* (CVPR 2025) — Dynamic Feature Modulator for selective G-buffer encoding

### Path C: Fine-Tuned Diffusion with G-Buffer Conditioning (Highest Artistic Quality)

**Keep diffusion but add depth/normals as native input channels.**

Instead of bolting on ControlNet (1.4GB per condition), expand the UNet's input channels to natively accept G-buffer data. This is how SD-Inpainting was created (4→9 channels).

**What it looks like:**
```
N64 frame → VAE encode ──┐
Z-buffer depth (encoded) ─┼→ Modified SD-Turbo UNet (5-8 input channels) → enhanced frame
Computed normals ──────────┘   (LoRA fine-tuned, LCM-distilled for 1-2 steps)
```

**How to do it:**
1. Load SD-Turbo UNet, change `in_channels` from 4 to 8 (4 latent + 1 depth + 3 normals)
2. Zero-initialize the new channel weights (so it starts from pretrained behavior)
3. LoRA fine-tune (rank 16-64) on paired N64 frames with G-buffer data
4. Apply LCM-LoRA for 1-2 step inference
5. Convert to TensorRT

**VRAM for training:** ~8GB with LoRA rank 16 + gradient checkpointing + mixed precision.
**VRAM for inference:** ~4GB (same as current, just different input shape).
**FPS:** Similar to current (~39 FPS with TensorRT).

**Alternative: Use T2I-Adapters instead of ControlNet**
- T2I-Adapter is only ~77M params vs ControlNet's ~1.4B
- Stack 2-3 adapters (depth + edge + color) for ~300MB total
- **Uni-ControlNet** (NeurIPS 2023) handles 7+ condition types with just 2 adapters

**Key papers:**
- *RGB-X* (Adobe, SIGGRAPH 2024) — fine-tune SD for bidirectional intrinsic decomposition
- *Diffusion-based G-buffer Generation and Rendering* (2025) — multi-channel CNN feature extractor → ControlNet
- *Uni-ControlNet* (NeurIPS 2023) — one model, many conditions, two adapters

---

## Recommended Strategy

**Phase 1 — Immediate (Path A):** Drop in `realesr-animevideov3` + TensorRT as an alternative rendering mode. This gives us a 60+ FPS baseline with dramatically less VRAM, proving the pipeline works at speed. Keep the existing diffusion path as a "creative mode."

**Phase 2 — Short-term (Path B start):** Add screen-space normals computation from Z-buffer (3 lines of code). Build a training data pipeline: run SM64 at native res + 4x upscaled, capture paired frames with depth + normals. This data is useful for both Path B and Path C.

**Phase 3 — Medium-term (Path B full):** Train a custom 2-5M param CNN with RGB + depth + normals + temporal input. Use GameIR dataset for pre-training, fine-tune on N64 pairs. Deploy with TensorRT for 60-120 FPS real-time enhancement.

**Phase 4 — Exploration (Path C):** For users who want artistic styles (Ghibli, watercolor, etc.), fine-tune the diffusion path with proper G-buffer channel expansion. This path is slower but produces creative output that a CNN upscaler can't.

---

## Key References

### G-Buffer Conditioned Neural Rendering
- [Neural Supersampling for Real-Time Rendering](https://dl.acm.org/doi/abs/10.1145/3386569.3392376) — Meta, SIGGRAPH 2020
- [NSRD: Neural SR with Radiance Demodulation](https://arxiv.org/abs/2308.06699) — CVPR 2024
- [QRISP: Efficient Neural Supersampling](https://arxiv.org/abs/2308.01483) — ICCV 2023
- [Efficient Video SR with Decoupled G-buffer Guidance](https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_Efficient_Video_Super-Resolution_for_Real-time_Rendering_with_Decoupled_G-buffer_Guidance_CVPR_2025_paper.pdf) — CVPR 2025
- [DLSS 4 Transformer Architecture](https://research.nvidia.com/labs/adlr/DLSS4/) — NVIDIA

### Lightweight Game Upscaling
- [GameSR: Real-Time Super-Resolution for Interactive Gaming](https://openreview.net/forum?id=wnJkdo5Gu9)
- [LCS: AI-based Low-Complexity Scaler](https://arxiv.org/abs/2507.22873) — July 2025
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [NTIRE 2025 Efficient SR Challenge](https://arxiv.org/html/2504.10686v1)
- [STSSNet: Low-latency Space-time Supersampling](https://arxiv.org/abs/2312.10890) — AAAI 2024

### N64 Emulator Internals
- [N64brew Wiki — RDP Pipeline](https://n64brew.dev/wiki/Reality_Display_Processor/Pipeline)
- [RT64 — Modern N64 Renderer](https://github.com/rt64/rt64)
- [parallel-rdp — Vulkan Compute RDP](https://github.com/Themaister/parallel-rdp)
- [angrylion-rdp-plus — Reference Software RDP](https://github.com/ata4/angrylion-rdp-plus)
- [N64 Normals and Lighting](https://www.moria.us/blog/2020/11/n64-part18-normals-and-lighting)
- [Normal Reconstruction from Depth](https://atyuwen.github.io/posts/normal-reconstruction/)

### Multi-Channel Conditioning
- [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) — ~77M params, composable
- [Uni-ControlNet](https://github.com/ShihaoZhaoZSH/Uni-ControlNet) — NeurIPS 2023, 2 adapters for 7+ conditions
- [RGB-X: Intrinsic Decomposition](https://arxiv.org/abs/2405.00666) — Adobe, SIGGRAPH 2024
- [HuggingFace: Adapting Model Input Channels](https://huggingface.co/docs/diffusers/training/adapt_a_model)

### Training Data
- [GameIR Dataset](https://huggingface.co/datasets/LLLebin/GameIR) — 19,200 paired game frames with depth
- [SRGD Dataset](https://github.com/epishchik/SRGD) — 14K frames from UE at 4 resolutions
- [BasicSR Training Framework](https://github.com/XPixelGroup/BasicSR)
