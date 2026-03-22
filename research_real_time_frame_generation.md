# Real-Time Image-to-Image Frame Generation Models: Comprehensive Research

**Target hardware:** NVIDIA GTX 4060 (8GB VRAM)
**Input resolution:** ~320x240 (N64 native), output 640x480 or 720p
**Target framerate:** 24+ FPS (ideally 30+ FPS)
**Date:** March 2026

---

## Executive Summary

For real-time frame generation on an 8GB RTX 4060, the most viable approaches in order of feasibility are:

1. **GAN-based approaches (HyPER-GAN, compressed pix2pix)** - Fastest inference (<15ms/frame at 720p), lowest VRAM (<1GB), most practical for real-time
2. **StreamDiffusion + SD-Turbo** - Proven ~10ms/frame at 512x512, designed for real-time img2img pipelines
3. **Frame interpolation (RIFE with TensorRT)** - Proven real-time at 720p (45+ FPS on RTX 3050), complementary to generation
4. **pix2pix-turbo (one-step diffusion)** - ~290ms at 512x512 on A6000, likely ~150-200ms on RTX 4060, too slow for 24fps alone but possible at lower resolution
5. **LCM/SD-Turbo standalone** - Feasible at 2-4 steps with TensorRT, borderline real-time

---

## 1. pix2pix-turbo (One-Step Diffusion)

**Paper:** "One-Step Image Translation with Text-to-Image Models" (March 2024)
**arXiv:** https://arxiv.org/abs/2403.12036
**Code:** https://github.com/GaParmar/img2img-turbo
**Open source:** Yes

### Architecture
- Built on SD-Turbo (Stable Diffusion Turbo), adapting it for paired/unpaired image-to-image translation
- Consolidates encoder, UNet, and decoder into a single end-to-end generator
- Small trainable weights added; preserves input image structure
- Uses adversarial learning objectives for fine-tuning to new domains

### Inference Speed
| Resolution | GPU | Time |
|-----------|-----|------|
| 512x512 | A6000 | 290ms |
| 512x512 | A100 | 110ms |

- Single step inference (matches 50-step ControlNet quality)
- **Estimated on RTX 4060:** ~200-300ms at 512x512 (3-5 FPS) - NOT real-time at this resolution
- At 320x240 input, speed would improve ~4x due to fewer pixels = potentially 12-20 FPS

### VRAM
- Based on SD-Turbo (~3-4 GB at 512x512 in fp16)
- Fits in 8GB VRAM

### Conditional img2img
- Yes, this is specifically designed for conditional image-to-image translation
- Supports sketch-to-image, day-to-night, and arbitrary domain translation
- CycleGAN-Turbo variant supports unpaired training

### TensorRT/ONNX
- Quantized FLUX variant exists (4-bit via Nunchaku)
- OpenVINO integration documented
- TensorRT compilation would further accelerate

### Verdict for N64 DLSS
Could work at native N64 resolution (320x240) but likely 10-20 FPS without heavy optimization. Not ideal for real-time without TensorRT compilation and resolution constraints.

---

## 2. StreamDiffusion

**Paper:** "StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation" (Dec 2023, ICCV 2025)
**arXiv:** https://arxiv.org/abs/2312.12491
**Code:** https://github.com/cumulo-autumn/StreamDiffusion
**Open source:** Yes

### Architecture
- Pipeline-level optimization for diffusion models, NOT a new model
- Key innovations:
  - **Stream Batch:** Batches denoising steps across frames (1.5x speedup)
  - **Residual CFG (R-CFG):** Approximates negative condition (up to 2.05x speedup)
  - **Stochastic Similarity Filter (SSF):** Skips redundant computation for similar frames
  - **Input/output queues** for smooth data flow
  - **Tiny VAE** (madebyollin/taesd) for faster encoding/decoding

### Inference Speed
| Model | Steps | Resolution | GPU | FPS |
|-------|-------|-----------|-----|-----|
| SD-Turbo | 1 | 512x512 | RTX 4090 | ~91-93 FPS |
| LCM-LoRA+KohakuV2 | 4 | 512x512 | RTX 4090 | ~37 FPS |
| SD-Turbo | 1 | 512x512 | RTX 4090 | ~10ms/frame |

- **59.6x faster** than Diffusers AutoPipeline
- On RTX 3060: 2.39x energy reduction for static scenes

### VRAM
- At 512x832 with batch size 3: ~9GB runtime, ~12GB during TensorRT build
- At 512x512 with batch size 1: likely ~4-6GB
- **TensorRT build may peak above 8GB** on RTX 4060 - this is a known issue

### Conditional img2img
- Yes, supports img2img mode natively
- Example code at `examples/img2img/single.py`
- Screen-to-image variant exists (ScreenDiffusion)

### TensorRT Support
- Built-in TensorRT acceleration support
- Initial TensorRT engine build takes 5-10 minutes, cached after that
- Known VRAM issues during build on 8GB GPUs (Issue #161)

### Verdict for N64 DLSS
**STRONG CANDIDATE.** With SD-Turbo backend at 512x512, could achieve 30-50+ FPS on RTX 4060. The 8GB VRAM limit is tight for TensorRT builds but may work at 512x512 or lower. At 320x240 input upscaled to 512x512 output, this is likely the best diffusion-based approach.

---

## 3. SDXL Turbo / SD Turbo

**Model:** Stability AI
**HuggingFace:** https://huggingface.co/stabilityai/sdxl-turbo
**TensorRT variant:** https://huggingface.co/stabilityai/sdxl-turbo-tensorrt
**Open source:** Yes (model weights available)

### Architecture
- Uses Adversarial Diffusion Distillation (ADD)
- SDXL Turbo: 3.1 billion parameters, 512x512 native
- SD Turbo: Smaller, based on SD 2.1, also 512x512
- Single-step generation by design (no CFG needed)

### Inference Speed
| Model | Steps | Resolution | GPU | Speed |
|-------|-------|-----------|-----|-------|
| SDXL Turbo | 1 | 512x512 | A100 | ~207ms total (67ms UNet) |
| SD Turbo | 1 | 512x512 | A100 | Faster than SDXL Turbo |
| SDXL Turbo | 1 | 512x512 | RTX 4090 | ~4 images/sec with TensorRT |

- ONNX Runtime: up to 229% speedup (SDXL Turbo), 120% (SD Turbo) vs PyTorch

### VRAM
- **SDXL Turbo:** Recommends 12+ GB VRAM. Tight on 8GB.
- **SD Turbo:** ~4-5 GB at 512x512 fp16. **Fits well in 8GB.**
- No CFG required = lower memory than standard SD

### Conditional img2img
- Supports img2img with `num_inference_steps * strength >= 1`
- Example: strength=0.5, steps=2 = 1 effective step

### TensorRT/ONNX
- Official TensorRT weights available for SDXL Turbo
- ONNX Runtime with TensorRT EP provides best performance
- SD Turbo is the better choice for 8GB VRAM

### Verdict for N64 DLSS
**SD Turbo is a good candidate** for the backbone model. Use it through StreamDiffusion for best results. SDXL Turbo is too large for 8GB VRAM without careful optimization. SD Turbo at 512x512 with 1 step should achieve 15-30+ FPS on RTX 4060 with TensorRT.

---

## 4. LCM (Latent Consistency Models)

**Paper:** "Latent Consistency Models: Synthesizing High-Resolution Images with Few-step Inference" (Oct 2023)
**arXiv:** https://arxiv.org/abs/2310.04378
**Code:** https://github.com/luosiallen/latent-consistency-model
**LCM-LoRA:** https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
**Open source:** Yes

### Architecture
- Distilled from Stable Diffusion via consistency distillation
- Predicts the solution of augmented probability flow ODE directly
- LCM-LoRA: Plug-and-play adapter (no full model retraining needed)
- Works with SD 1.5, SDXL, SSD-1B

### Inference Speed
- 2-4 steps for high-quality output (vs 25-50 for standard SD)
- 10-100x reduction in runtime vs classic diffusion
- With TensorRT: ~9x faster than standard 50-step pipeline
- On RTX 4090: ~37 FPS when combined with StreamDiffusion (4 steps, 512x512)

### VRAM
- Same as base model (SD 1.5: ~3-4GB, SDXL: ~6-8GB in fp16)
- LCM-LoRA adds minimal overhead

### Conditional img2img
- Yes, img2img supported
- Works with standard diffusers img2img pipeline
- guidance_scale=1 recommended (no CFG needed)

### TensorRT/ONNX
- TensorRT acceleration available through NVIDIA's SD WebUI extension
- Real-Time LCM demo exists: https://github.com/radames/Real-Time-Latent-Consistency-Model

### Verdict for N64 DLSS
**Good candidate via StreamDiffusion.** LCM-LoRA + SD 1.5 is lightweight and fast. At 4 steps through StreamDiffusion, expect 15-25 FPS on RTX 4060 at 512x512. Pair with TensorRT for best results.

---

## 5. Consistency Models (OpenAI)

**Paper:** "Consistency Models" (March 2023)
**arXiv:** https://arxiv.org/abs/2303.01469
**Scaled CMs (2024):** https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/
**Code:** https://github.com/openai/consistency_models
**Open source:** Yes (original), scaled versions partially

### Architecture
- Maps noise directly to data in a single step
- No iterative denoising required
- Scaled Consistency Models (sCM): 1.5B parameters, 512x512 ImageNet

### Inference Speed
- sCM (1.5B params): 0.11 seconds on A100 for a single sample (2 steps)
- ~50x wall-clock speedup over equivalent diffusion models
- Quality comparable to diffusion models with just 2 sampling steps

### VRAM
- 1.5B model: likely ~3-4GB in fp16
- Original smaller models: <2GB

### Conditional img2img
- Original models: primarily unconditional/class-conditional (ImageNet)
- Not designed for arbitrary img2img translation
- Would require custom training/fine-tuning for game frame generation

### Limitations
- Not as easily adapted for img2img as SD-based models
- Ecosystem and tooling much smaller than Stable Diffusion family
- Research-oriented, less production-ready

### Verdict for N64 DLSS
**Not directly applicable.** These are primarily unconditional generators. The SD-Turbo and LCM families inherited the key ideas and are more practical for img2img applications.

---

## 6. Frame Interpolation: RIFE / FILM / AMT

### RIFE (Real-Time Intermediate Flow Estimation)

**Paper:** ECCV 2022
**Code:** https://github.com/hzwer/ECCV2022-RIFE
**TensorRT:** https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt
**NCNN/Vulkan:** https://github.com/nihui/rife-ncnn-vulkan
**Open source:** Yes

#### Performance
| Resolution | GPU | Backend | FPS |
|-----------|-----|---------|-----|
| 720p (2x interp) | RTX 2080 Ti | PyTorch | 30+ FPS |
| 1080p | RTX 3050 | NCNN | 30.71 FPS |
| 1080p | RTX 3050 | TensorRT | 45.91 FPS |
| 1080p | RTX 4090 | TensorRT | 288 FPS |

- TensorRT is up to 100% faster than NCNN/Vulkan
- At 320x240 or 480p: easily 100+ FPS on RTX 4060

#### VRAM
- Very lightweight: ~500MB-1GB
- Runs on mobile devices via NCNN/Vulkan

### FILM (Frame Interpolation for Large Motion)
- Google Research, better quality for large motion
- Slower than RIFE (5-10x slower)
- Not suited for real-time

### AMT (All-Pairs Multi-Field Transforms)
**Paper:** CVPR 2023
**Code:** https://github.com/MCG-NKU/AMT
- Convolution-based, competes with Transformer models
- Lightweight and efficient
- State-of-the-art on multiple benchmarks

### Verdict for N64 DLSS
**HIGHLY RELEVANT as a complement.** RIFE with TensorRT can interpolate frames at 100+ FPS at N64 resolutions. Strategy: generate every other frame with AI, interpolate the rest with RIFE. This could effectively double your output framerate. RIFE is proven, lightweight, and real-time.

---

## 7. GameNGen (Google)

**Paper:** "Diffusion Models Are Real-Time Game Engines" (August 2024, ICLR 2025)
**arXiv:** https://arxiv.org/abs/2408.14837
**Website:** https://gamengen.github.io/
**Community reproduction:** https://github.com/arnaudstiegler/gameNgen-repro
**Open source:** Official weights NOT released. Unofficial reproductions exist.

### Architecture
- Based on Stable Diffusion v1.4 (fine-tuned)
- Two-phase training:
  1. RL agent plays DOOM, records gameplay sessions
  2. Diffusion model trained to predict next frame conditioned on past frames + actions
- Uses only 4 DDIM denoising steps per frame
- Conditioning augmentations for stable autoregressive generation
- Noise augmentation prevents quality degradation over long trajectories

### Inference Speed
- **20 FPS on a single TPU** (Google TPU v5)
- Total inference cost: ~50ms per frame (40ms UNet + 10ms autoencoder)
- Single-step distillation could reach 50 FPS but with quality degradation
- Resolution: 320x240 (DOOM native)

### VRAM
- Based on SD 1.4: ~3-4 GB in fp16
- Would fit in 8GB VRAM on consumer GPU
- TPU-optimized; GPU performance may differ

### Key Insight for N64 Project
GameNGen proves that a fine-tuned SD 1.4 with 4 denoising steps can render game frames at 20 FPS at 320x240 resolution. This is EXACTLY the resolution and framerate target for N64 emulation. The approach is directly relevant.

### Limitations
- Official weights not public
- Trained specifically on DOOM gameplay
- Requires game-specific training data (RL agent recordings)
- TPU-optimized (GPU performance uncertain but likely similar with TensorRT)

### Verdict for N64 DLSS
**MOST RELEVANT RESEARCH.** Proves the concept works at exactly the right resolution (320x240) and framerate (20 FPS). Adapting this approach to N64 games would require: (1) collecting training data from N64 gameplay, (2) fine-tuning SD 1.4 on that data, (3) optimizing with TensorRT for consumer GPU. The 4-step approach with SD 1.4 is achievable on RTX 4060.

---

## 8. Other Neural Game Engines

### DIAMOND (DIffusion As a Model Of eNvironment Dreams)
**Paper:** NeurIPS 2024 Spotlight
**Code:** https://github.com/eloialonso/diamond
**Open source:** Yes

- Diffusion world model for Atari games and Counter-Strike
- Atari models: 4M parameters (tiny!)
- CS:GO model: 381M parameters (including 51M upsampler)
- Runs at **10 Hz (10 FPS) on RTX 3090**
- Trained on 87 hours of CS:GO gameplay

### Oasis (Decart AI + Meta)
**Code:** https://github.com/etched-ai/open-oasis (500M model)
**Website:** https://oasis-model.github.io/
**Open source:** Yes (500M variant)

- Diffusion Transformer (DiT) + ViT auto-encoder
- Generates Minecraft-like gameplay at **20 FPS**
- Each frame generated in ~40ms
- 500M parameter open-source model available
- Full model optimized for Etched Sohu ASIC

---

## 9. GAN-Based Approaches (Fastest Option)

### Original pix2pix
**Code:** https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

- Single forward pass through generator
- At 256x256: ~5-15ms on modern GPU (estimated)
- At 512x512: ~20-40ms on consumer GPU
- Very lightweight generator (UNet-based, ~50-100M params)
- VRAM: <2GB

### GAN Compression (MIT HAN Lab)
**Paper:** CVPR 2020
**Code:** https://github.com/mit-han-lab/gan-compression

- Compresses pix2pix by 12x computation
- Compressed model achieves **40 FPS** on mobile devices
- **~21ms latency on iPhone 14 Pro** (ResNet-based pix2pix)
- 30 FPS on iPhone 14 with efficient on-device model

### HyPER-GAN (2026) -- MOST RELEVANT GAN
**Paper:** https://arxiv.org/abs/2603.10604
**Code:** https://github.com/stefanos50/HyPER-GAN
**Open source:** Yes (code + pretrained models)

| Resolution | GPU | Latency | FPS | VRAM |
|-----------|-----|---------|-----|------|
| 720p | RTX 4070 Super | 12.3ms | 81 FPS | 0.8 GB |
| 1080p | RTX 4070 Super | 29.6ms | 33.7 FPS | 1.5 GB |

- **Compact U-Net generator**
- Designed specifically for game rendering photorealism enhancement
- Trained on GTA-V, tested on CARLA simulator and other games
- Uses ~50% less VRAM than competing methods
- Paired training with hybrid real-world patches

### REGEN (2025)
**Paper:** https://arxiv.org/abs/2508.17061
**Code:** https://github.com/stefanos50/REGEN
**Open source:** Yes

- Dual-stage framework: unpaired Im2Im -> paired lightweight model
- 32x faster than robust unpaired method on GTA V
- 12x faster on Unreal Engine
- No low-level engine access required

### E2GAN (Efficient Training of Efficient GANs, 2024)
**Paper:** https://arxiv.org/html/2401.06127v1

- Efficient training and inference for image-to-image translation
- Focus on reducing both training cost and inference time

### Verdict for N64 DLSS
**GAN approaches are the fastest option.** HyPER-GAN achieves 81 FPS at 720p using only 0.8GB VRAM on an RTX 4070 Super. On an RTX 4060, expect ~60-70 FPS at 720p. This is more than enough for real-time frame generation. The challenge is training quality -- GANs produce less diverse/creative outputs than diffusion models, but for style transfer on game frames, this is actually desirable (consistency matters more than diversity).

---

## 10. TensorRT Optimization

### Key Performance Gains
| Optimization | Speedup |
|-------------|---------|
| TensorRT FP16 vs PyTorch | 1.5-2x |
| TensorRT INT8 vs PyTorch | 1.72x |
| TensorRT FP8 vs PyTorch | 1.95x |
| ONNX Runtime + TensorRT EP vs PyTorch | Up to 229% (SDXL Turbo) |
| TensorRT for RIFE vs NCNN | Up to 100% |

### SD UNet Single Step Latency with TensorRT
- SD v2.1 UNet at 512x512: ~43ms per step (TensorRT FP16)
- **Estimated SD 1.5 UNet at 320x240: ~10-15ms per step** (4x fewer pixels)
- With 4 steps (GameNGen approach): ~40-60ms total = 16-25 FPS at 320x240

### Practical Considerations for RTX 4060
- TensorRT engine build may require more VRAM than runtime inference
- Build once, cache for reuse
- FP16 is the sweet spot for RTX 4060 (has Tensor Cores)
- INT8 quantization possible but may reduce quality
- Engine build for StreamDiffusion at 512x512 may peak at 9-12GB (problematic for 8GB)

---

## 11. ONNX Runtime Optimization

### Key Features
- CUDA Execution Provider: Uses cuDNN, fast startup
- TensorRT Execution Provider: Optimizes full graph, slower startup but faster inference
- Flash Attention and Memory Efficient Attention
- NHWC tensor layout for Tensor Core GPUs

### Benchmark Results (SD Turbo / SDXL Turbo)
- Tested on A100 and RTX 4090
- SDXL Turbo: up to 229% throughput improvement vs PyTorch
- SD Turbo: up to 120% throughput improvement
- Dynamic shape support (512x512 to 768x768)
- Best performance with static shapes + TensorRT EP

### For RTX 4060
- ONNX Runtime with CUDA EP is the safest option (lower peak VRAM)
- TensorRT EP gives best performance but may hit VRAM limits during optimization
- fp16 precision recommended

---

## Recommended Architecture for N64 DLSS

### Option A: Fastest (GAN-Based)
```
N64 Frame (320x240) -> HyPER-GAN / Custom pix2pix -> Enhanced Frame (720p)
                                                        |
                                                   RIFE Interpolation -> Doubled framerate
```
- Expected: 60-80 FPS at 720p, <1GB VRAM
- Requires training custom GAN on N64 game data
- Most practical for real-time use

### Option B: Best Quality (Diffusion-Based)
```
N64 Frame (320x240) -> StreamDiffusion + SD-Turbo (1 step) -> Enhanced Frame (512x512)
                                                                |
                                                           RIFE Interpolation -> Doubled framerate
```
- Expected: 20-40 FPS at 512x512, ~4-6GB VRAM
- Better visual quality and style transfer capability
- Uses StreamDiffusion's batching and R-CFG optimizations

### Option C: GameNGen-Style (Frame Prediction)
```
Previous Frames + Actions -> Fine-tuned SD 1.4 (4 steps) -> Next Frame (320x240)
                                                              |
                                                         Super Resolution -> 720p
```
- Expected: 15-25 FPS at 320x240, ~4GB VRAM
- Most creative/generative approach
- Requires significant training data collection
- Can predict frames ahead (latency hiding)

### Option D: Hybrid (Recommended)
```
N64 Frame (320x240) -> Lightweight GAN (style transfer, ~5ms) -> Styled Frame (640x480)
                                                                    |
                                                               RIFE (frame interp, ~3ms) -> 2x frames
                                                                    |
                                                               Real-ESRGAN compact (optional upscale)
```
- Expected: 100+ FPS pipeline, <2GB VRAM
- Each component is proven and lightweight
- Can be optimized independently
- RIFE + GAN combo gives best speed/quality tradeoff

---

## Summary Table

| Model | Type | Inference/frame | Resolution | VRAM | img2img | Open Source | Real-time on 4060? |
|-------|------|----------------|-----------|------|---------|-------------|-------------------|
| HyPER-GAN | GAN | 12ms | 720p | 0.8GB | Yes | Yes | YES (81 FPS on 4070S) |
| StreamDiffusion+SD-Turbo | Diffusion | ~10ms | 512x512 | 4-6GB | Yes | Yes | YES (~30-50 FPS est.) |
| pix2pix (compressed) | GAN | ~21ms | 256x256 | <1GB | Yes | Yes | YES (40+ FPS) |
| RIFE (TensorRT) | Interpolation | ~3-5ms | 720p | <1GB | N/A | Yes | YES (45+ FPS on 3050) |
| pix2pix-turbo | Diffusion | 290ms | 512x512 | 3-4GB | Yes | Yes | NO (3-5 FPS) |
| SD Turbo (standalone) | Diffusion | ~100-200ms | 512x512 | 3-4GB | Yes | Yes | BORDERLINE (5-10 FPS) |
| SDXL Turbo | Diffusion | ~200ms+ | 512x512 | 8+GB | Yes | Yes | NO (VRAM limited) |
| LCM (4 steps) | Diffusion | ~100-200ms | 512x512 | 3-4GB | Yes | Yes | BORDERLINE |
| GameNGen | Diffusion | 50ms | 320x240 | 3-4GB | Yes (frame prediction) | No (unofficial repros) | LIKELY (with TensorRT) |
| DIAMOND | Diffusion | 100ms | Atari/low | <2GB | Yes (world model) | Yes | YES (10 FPS proven) |
| Oasis 500M | DiT | 40ms | Low | ~2-4GB | Yes (world model) | Yes (500M) | POSSIBLE |
| Consistency Models | Diffusion | 110ms | 512x512 | 3-4GB | Limited | Partial | BORDERLINE |
| SDXL Lightning | Diffusion | ~200ms+ | 1024x1024 | 12+GB | Yes | Yes | NO (VRAM) |
| Real-ESRGAN | SR (GAN) | ~83ms | 4x upscale | 2-4GB | Super-res only | Yes | POSSIBLE at low res |

---

## Key Papers and Resources

1. **pix2pix-turbo:** https://arxiv.org/abs/2403.12036
2. **StreamDiffusion:** https://arxiv.org/abs/2312.12491
3. **GameNGen:** https://arxiv.org/abs/2408.14837
4. **DIAMOND:** https://arxiv.org/abs/2405.12399
5. **LCM:** https://arxiv.org/abs/2310.04378
6. **Consistency Models:** https://arxiv.org/abs/2303.01469
7. **RIFE:** https://github.com/hzwer/ECCV2022-RIFE
8. **HyPER-GAN:** https://arxiv.org/abs/2603.10604
9. **REGEN:** https://arxiv.org/abs/2508.17061
10. **GAN Compression:** https://github.com/mit-han-lab/gan-compression
11. **Oasis:** https://github.com/etched-ai/open-oasis
12. **E2GAN:** https://arxiv.org/html/2401.06127v1
13. **AMT:** https://arxiv.org/abs/2304.09790
14. **SDXL Turbo TensorRT:** https://huggingface.co/stabilityai/sdxl-turbo-tensorrt
15. **ONNX Runtime SD acceleration:** https://huggingface.co/blog/sdxl_ort_inference
16. **NVIDIA TensorRT for SD:** https://developer.nvidia.com/blog/new-stable-diffusion-models-accelerated-with-nvidia-tensorrt/
17. **Real-Time LCM Demo:** https://github.com/radames/Real-Time-Latent-Consistency-Model
