# Deep Research: Lightweight Real-Time Neural Upscaling for N64 Frame Enhancement

## Date: 2026-03-22

This document summarizes deep research into non-diffusion neural upscaling approaches
that could replace or complement the current StreamDiffusion + SD-Turbo pipeline,
targeting real-time performance on an 8GB VRAM GPU (GTX 4060).

---

## 1. Non-Diffusion Approaches for Real-Time Frame Enhancement

### 1.1 ESRGAN / Real-ESRGAN Family

**Architecture:** RRDBNet (Residual in Residual Dense Block Network)

| Model | Architecture | Params | Size | Speed (V100) | Notes |
|-------|-------------|--------|------|---------------|-------|
| RealESRGAN_x4plus | RRDBNet, 23 blocks | ~16.7M | ~64MB | Slow (~3-5 FPS at 640x480) | Best quality, not real-time |
| RealESRGAN_x4plus_anime_6B | RRDBNet, 6 blocks | ~4.5M | ~17MB | Moderate | Reduced color bleeding |
| realesr-general-x4v3 | SRVGGNetCompact, 32 conv | ~2.6M | ~16MB | ~30+ FPS at 640x480 | Adjustable denoise strength |
| realesr-animevideov3 | SRVGGNetCompact, 16 conv | ~1.3M | ~8MB | 65.9 FPS at 640x480, 22.6 FPS at 720p | **Best real-time candidate** |

**Auxiliary Data Conditioning:** ESRGAN/Real-ESRGAN do NOT natively support depth/normal
conditioning. To add auxiliary inputs, you would need to:
- Modify the input layer from 3 channels (RGB) to 3+N channels (RGB + depth + normals)
- Retrain or fine-tune from pre-trained weights (zero-initialize new input channels)
- Use A-ESRGAN (multi-scale attention U-Net discriminator) as a starting point

**Fine-tuning on Custom Data (BasicSR):**
- Two-stage training: (1) train Real-ESRNet with L1 loss, (2) add GAN + perceptual loss
- Config: `finetune_realesrgan_x4plus.yml` - modify dataset paths and meta info
- Can use paired LR-HR data or HR-only (with degradation synthesis)
- Community has extensively used this for game texture upscaling

### 1.2 Neural Supersampling (Pre-DLSS / Academic Work)

**Facebook/Meta Neural Supersampling (2020, Lei Xiao et al.):**
- Uses color + motion vectors + depth as input
- Temporal accumulation via recurrent architecture
- Targets 4x upscaling (540p -> 1080p)
- ~12ms inference on RTX 3090 at 1080p output

**Neural Super-Resolution with Radiance Demodulation (2023):**
- Architecture: ConvLSTM with U-shaped module and residual channel attention
- **1.61M parameters** (much smaller than BasicVSR++ at 7.32M)
- **12.41ms on RTX 3090** at 1920x1080 output
- Key innovation: separates rendered image into lighting and material components
- Uses depth and normals for motion estimation and occlusion detection
- HR material component (albedo) can be generated cheaply without global illumination
- Training: per-scene, 200 epochs in ~24 hours, 5000+ augmented patches (96x96 crops)

**AMD Joint Denoising and Upscaling (I3D 2025):**
- Multi-branch U-Net: separate radiance branch and guiding branch
- Auxiliary inputs: albedo, normal, roughness, depth, specular hit distance, motion vectors
- Guiding branch deliberately lightweight since G-buffers are clean (just aliased)
- Temporal accumulation with motion vector reprojection

**Arm Neural Super Sampling:**
- Four-level UNet backbone with skip connections
- Inputs: color, motion vectors, depth, luma derivative, disocclusion mask
- Output: per-pixel 4x4 filter kernel + temporal accumulation coefficients + hidden state
- Target: <=4ms per frame, ~10 GOPs computational budget
- Uses parameter prediction (quantization-friendly) rather than direct image prediction
- Training: 540p -> 1080p, ~100 frame sequences, PyTorch + ExecuTorch for QAT

### 1.3 Lightweight Transformer-Based Upscalers

**SwinIR / Swin2SR:**
- SwinIR lightweight: 0.89M params (60-dim features, 4 RSTB blocks, 6 STL, 6 heads)
- Full SwinIR: 11.8M params - up to 67% fewer params than IPT (115M)
- Swin2SR improves training stability with SwinV2 layers
- Not inherently real-time but could be with TensorRT optimization

**NGramSwin (CVPR 2023):**
- Uses N-Gram context (neighboring local windows in Swin)
- SwinIR-NG variant outperforms other lightweight SR methods
- Code: https://github.com/rami0205/NGramSwin

**SAFMN (ICCV 2023):**
- Spatially-Adaptive Feature Modulation Network
- 85% fewer params than CARN, 42% fewer than ShuffleMixer
- 316K fewer params than SwinIR-light, nearly 7x faster
- Code: https://github.com/sunny2109/SAFMN

**SPAN (CVPR 2024):**
- Swift Parameter-free Attention Network
- ~0.43M parameters, superior to RLFN with 50K fewer params
- Better inference speed AND quality vs. RLFN/RLFN-S
- Code: https://github.com/hongyuanyu/SPAN

### 1.4 Game-Specific GAN Upscalers

**GameSR (2024-2025):**
- Specifically designed for game frame upscaling
- Architecture: reparameterized conv blocks + PixelUnshuffle + lightweight ConvLSTM
- **Real-time: up to 240 FPS** on desktop GPU
- Mobile variant (GameSR-M): 26.56ms per 1080p frame on Adreno 750
- Reparameterization collapses 3 convolutions into 1 at inference time
- Without ConvLSTM: only 65K params, 3.05ms, but -5dB PSNR drop
- Engine-independent: works on encoded frames, no source code access needed
- Near-parity with DLSS/FSR without engine integration
- Paper: https://openreview.net/forum?id=wnJkdo5Gu9

**LCS - Low-Complexity Scaler (July 2025):**
- Built on DIPNet/RLFN architecture with RRFBs
- 4 RRFBs with 38 feature channels, ESA attention blocks
- Full model: 0.74M params, 89.5ms, 672 GMACs
- **Reparameterized: 0.21M params, 30.9ms, 175 GMACs**
- Quantized INT8: 0.21M params, 175 GMACs (even faster)
- Trained on GameIR dataset (CARLA/UE4 game-rendered pairs)
- Beats AMD FSR1 and EASF on perceptual metrics (NIQE 3.43 vs 5.55, LPIPS 0.150 vs 0.199)
- Paper: https://arxiv.org/abs/2507.22873

**CuNNy:**
- Convolutional upscaling neural network as GLSL shader
- Runs entirely on GPU shader pipeline (no CUDA/compute dependency)
- Variants: 8x32, 4x32, 4x16, 3x12, 2x12, fast, faster, veryfast
- NVL variant specifically trained for games and illustrations
- dp4a (INT8 dot product) acceleration for supported hardware
- 2x upscaling only
- Code: https://github.com/funnyplanter/CuNNy

**ArtCNN:**
- Simple SISR CNNs for anime/game content, implemented as mpv shaders
- Code: https://github.com/Artoriuz/ArtCNN

---

## 2. Models Designed for Game Content Upscaling

### 2.1 GameIR Dataset

The premier dataset for game-specific super-resolution training:
- **19,200 LR-HR paired frames** from 640 videos
- Rendered at 720p (LR) and 1440p (HR) using CARLA simulator (Unreal Engine 4)
- **Includes G-buffers**: segmentation maps and depth maps
- Available: https://huggingface.co/datasets/LLLebin/GameIR
- Paper: https://arxiv.org/abs/2408.16866

### 2.2 SRGD (Super Resolution Gaming Dataset)

- Collected using Unreal Engine
- 14,431 train + 3,600 test images (GameEngineData)
- 29,726 train + 7,421 test images (DownscaleData)
- 4 resolutions: 270p, 360p, 540p, 1080p
- Available: https://huggingface.co/datasets/epishchik/SRGD

**Benchmark results (4x, 270p->1080p):**

| Model | Type | PSNR | SSIM | LPIPS |
|-------|------|------|------|-------|
| Real-ESRGAN | GAN | 23.54 | 0.799 | 0.392 |
| EMT | Transformer | 24.54 | 0.823 | 0.389 |
| ResShift | Diffusion | 23.04 | 0.799 | 0.483 |

### 2.3 Frame Interpolation (Complementary)

**RIFE (Real-Time Intermediate Flow Estimation):**
- Can generate intermediate frames to boost effective framerate
- Supports tensor cores on newer GPUs
- SVP integration for real-time playback
- Complementary to upscaling: upscale spatially + interpolate temporally

---

## 3. Training Custom Models with Auxiliary Inputs

### 3.1 Architecture Design for (LR frame + depth + normals) -> HR frame

**Recommended approach based on literature:**

```
Input: [RGB_LR (3ch) + Depth (1ch) + Normals (3ch)] = 7 channels
       OR use separate branches (recommended by AMD/Arm research)

Option A: Single-branch (simple)
- Modify input conv layer of SRVGGNetCompact from 3->7 channels
- Initialize new channels with zeros, fine-tune from pre-trained weights
- Fast to implement but may not optimally leverage auxiliary data

Option B: Dual-branch (better, based on AMD's approach)
- Main branch: processes RGB with full-depth feature extraction
- Guiding branch: lightweight layers for depth+normals (clean but aliased)
- Merge via feature concatenation or attention at bottleneck
- Auxiliary data is clean -> shallow processing is sufficient

Option C: Parameter prediction (best, based on Arm NSS)
- Instead of predicting HR image directly, predict per-pixel kernels
- Predict 4x4 upsampling kernels + temporal blend weights
- More quantization-friendly and bandwidth-efficient
- Output = apply_kernels(LR_input, predicted_kernels)
```

### 3.2 Dataset Creation Using N64 Emulator

**Key insight from RT64 architecture:**

RT64 uses deferred frame processing where entire frames are captured before GPU
submission. The WorkloadQueue operates on a dedicated render thread, providing access to:

- **Color framebuffer** at configurable resolution (native LR and upscaled HR)
- **Depth buffer** (Z-buffer already extracted in current project from RDRAM)
- **Vertex-level data**: transformed positions, normals from lighting calculations
- **Texture data**: original textures, UV coordinates, tile configurations
- **Projection/view matrices** per framebuffer pair

**Training data generation strategy:**

1. **Paired resolution rendering:**
   - Render each frame at native N64 resolution (320x240) -> LR input
   - Render same frame at 4x resolution (1280x960) via paraLLEl RDP upscaling -> HR target
   - Both are pixel-accurate since they use the same scene state

2. **Auxiliary buffer extraction:**
   - Z-buffer: already available from RDRAM (current project does this)
   - Depth map: linearize Z-buffer values using N64's depth range
   - Surface normals: can be approximated from depth map via gradient computation,
     or extracted from RT64's per-vertex lighting normals
   - Motion vectors: compute from consecutive frame projection matrices + vertex positions

3. **Practical capture pipeline:**
   ```python
   for each frame:
       # Run emulator at native resolution
       lr_frame = capture_framebuffer(320, 240)
       depth = extract_zbuffer_from_rdram()

       # Run same frame state at high resolution
       hr_frame = capture_framebuffer(1280, 960)  # or 2560x1920

       # Compute auxiliary data
       normals = compute_normals_from_depth(depth)

       save_training_pair(lr_frame, depth, normals, hr_frame)
   ```

4. **Data augmentation:**
   - Random crops (96x96 or 128x128 patches)
   - Horizontal flips, random rotations
   - Capture diverse game content: varied scenes, lighting, textures
   - Multiple N64 games for generalization

### 3.3 Transfer Learning Strategy

**Recommended pipeline:**

1. Start with pre-trained `realesr-animevideov3` (SRVGGNetCompact, 16 conv, ~1.3M params)
2. Modify input channels: 3 -> 7 (RGB + depth + normals + alpha/mask)
3. Zero-initialize new input channel weights
4. Stage 1: Fine-tune on GameIR dataset (19K pairs with depth maps) with L1 loss, 50 epochs
5. Stage 2: Fine-tune on N64-specific paired data with L1 + perceptual + adversarial loss
6. Stage 3: Quantize to FP16 and export to TensorRT

**Alternative starting points:**
- SPAN (0.43M params) - newer, competitive with more params
- RLFN (0.32M params) - proven real-time performance
- LCS architecture (0.21M reparameterized) - specifically designed for game content

---

## 4. Quantization and Optimization

### 4.1 FP16 Quantization

- Real-ESRGAN already defaults to FP16 inference
- Typical speedup: 1.5-2x over FP32 on consumer GPUs
- Negligible quality loss for super-resolution tasks
- All models listed above are FP16-compatible

### 4.2 INT8 Quantization

**Post-Training Quantization (PTQ):**
- TensorRT implicit quantization: automatically picks INT8 where faster
- Requires calibration dataset (~500-1000 representative images)
- Can reduce VRAM by ~2x and speed by ~1.3-1.7x over FP16
- WARNING: INT8 is not always faster than FP16 (depends on layer types and GPU arch)

**Quantization-Aware Training (QAT):**
- Better quality preservation than PTQ
- LCS paper uses Brevitas library for INT8 QAT
- Arm NSS uses ExecuTorch for QAT
- Recommended for final deployment model

### 4.3 TensorRT Optimization Pipeline

```
PyTorch model
    -> torch.onnx.export() -> ONNX model
    -> trtexec --fp16 (or --int8 with calibration) -> TensorRT engine
    -> Load via tensorrt Python API -> Real-time inference
```

**Key optimizations:**
- Layer fusion: conv + bias + ReLU -> single kernel (reduces memory bandwidth)
- Fixed input shapes: much faster than dynamic shapes
- Batch size 1 for real-time (no batching latency)
- For batch multiples of 32: best Tensor Core utilization
- Typical speedup: 2-5x over native PyTorch, up to 40x over CPU

### 4.4 Knowledge Distillation

**Multi-Teacher Knowledge Distillation (MTKD, ECCV 2024):**
- Train large teacher models (HAT, SwinIR, Real-ESRGAN)
- Distill into compact student (SPAN, RLFN, SRVGGNetCompact)
- Wavelet-based loss in both spatial and frequency domains
- Adaptive weighting: student prioritizes most informative teacher
- Paper: https://arxiv.org/abs/2404.09571

**Practical approach for this project:**
1. Train full Real-ESRGAN (23 blocks) on N64 game pairs -> teacher
2. Train SRVGGNetCompact (16 conv) to match teacher outputs -> student
3. Student loss = alpha * L1(student, HR) + beta * L1(student_features, teacher_features)

---

## 5. NTIRE 2025 Efficient SR Challenge - State of the Art

The most recent competitive benchmarks for efficient super-resolution (4x upscaling):

| Team | Params | FLOPs | Runtime | Approach |
|------|--------|-------|---------|----------|
| EMSR (winner) | 0.131M | 8.54G | 9.99ms | ConvLora + knowledge distillation |
| XiaomiMM | 0.148M | 9.68G | 9.55ms | Parameter-free attention + SPAN |
| ShannonLab | 0.172M | 11.23G | **8.62ms** | Optimized ECB + reparameterization |
| VPEG_C | **0.044M** | **3.13G** | 50.5ms | Dual attention + depthwise conv |
| HannahSR | 0.060M | 3.75G | 49.9ms | Multi-level refinement + reparam |
| XUPTBoys | 0.072M | 3.39G | 42.8ms | Frequency-guided hierarchical attn |

All maintained PSNR >= 26.90dB on validation. These represent the bleeding edge of
efficient SR, with sub-10ms inference at 4x upscaling.

---

## 6. Real-Time SR Benchmarks on Consumer GPUs

From the February 2026 streaming SR paper (RTX 2080, 720p->1440p 2x upscaling):

| Model | Params | FPS (RTX 2080) |
|-------|--------|----------------|
| Bicubic | - | 1829 |
| AsConvSR | ~0.03M | 213 |
| ESPCN | 0.04M | 201 |
| **EfRLFN** | **0.37M** | **271** |
| RLFN | 0.40M | 225 |
| SPAN | 0.43M | 60 |
| NVIDIA VSR | - | 52 |

With TensorRT FP16 optimization (360x480 input):
- EfRLFN: 82.8 FPS (2x), 68.2 FPS (4x)

Real-time threshold defined as >= 30 FPS. All models exceed this on RTX 2080.

**Implication for GTX 4060:** The 4060 is roughly comparable to the RTX 2080 in
compute throughput but has newer tensor cores and better FP16/INT8 support. At N64's
320x240 input resolution, even SPAN would likely achieve >100 FPS since the input
is much smaller than the 720p benchmark.

---

## 7. Recommended Approach for This Project

### Option A: Quick Win (replace diffusion, keep it simple)
- Use `realesr-animevideov3` (SRVGGNetCompact, 16 conv, 1.3M params, ~8MB)
- Export to TensorRT FP16
- Expected: >60 FPS at 320x240 -> 1280x960 on GTX 4060
- No auxiliary data, no custom training, works immediately
- VRAM: <500MB for the model

### Option B: Game-Optimized (best quality/speed tradeoff)
- Use GameSR or LCS architecture (reparameterized, <0.25M params)
- Fine-tune on GameIR dataset + custom N64 captured pairs
- Add temporal consistency via lightweight ConvLSTM
- Export to TensorRT FP16
- Expected: >120 FPS, near-DLSS quality
- VRAM: <200MB for the model

### Option C: Full Custom (maximum quality with auxiliary data)
- Design dual-branch network:
  - RGB branch: SRVGGNetCompact backbone
  - Guiding branch: 3-4 lightweight conv layers for depth + normals
  - Feature fusion via concatenation + 1x1 conv
  - Optional: ConvLSTM for temporal consistency
- Train on GameIR (has depth maps) + custom N64 pairs
- Knowledge distillation from full Real-ESRGAN teacher
- INT8 quantization via Brevitas QAT
- Export to TensorRT
- Expected: 60-120 FPS with superior quality
- VRAM: <500MB for the model

### VRAM Budget (GTX 4060, 8GB)
- Current SD-Turbo pipeline: ~4-6GB VRAM
- Option A: ~0.5GB (frees 7.5GB for emulator + display)
- Option B: ~0.2GB
- Option C: ~0.5GB
- All options leave massive VRAM headroom compared to current diffusion approach

---

## 8. Key References

### Papers
- GameSR: Real-Time Super-Resolution for Interactive Gaming (2024)
- LCS: AI-based Low-Complexity Scaler for Game Content (arXiv 2507.22873, July 2025)
- GameIR: Large-Scale Synthesized Ground-Truth Dataset (arXiv 2408.16866, ACCV 2024)
- Neural Super-Resolution with Radiance Demodulation (arXiv 2308.06699, 2023)
- MTKD: Multi-Teacher Knowledge Distillation for SR (ECCV 2024)
- PocketSR: SR Expert in Your Pocket (arXiv 2510.03012, October 2025)
- NTIRE 2025 Efficient SR Challenge Report (arXiv 2504.10686)
- Real-Time SR for Streaming Content (arXiv 2602.11339, February 2026)
- Taming HR Auxiliary G-Buffers for Deep Supersampling (IEEE TVCG, December 2025)
- AMD Joint Denoising and Upscaling (I3D 2025)
- Arm Neural Super Sampling (2024-2025)

### Code Repositories
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- BasicSR (training framework): https://github.com/XPixelGroup/BasicSR
- SPAN: https://github.com/hongyuanyu/SPAN
- SAFMN: https://github.com/sunny2109/SAFMN
- RLFN: https://github.com/bytedance/RLFN
- NGramSwin: https://github.com/rami0205/NGramSwin
- CuNNy: https://github.com/funnyplanter/CuNNy
- ArtCNN: https://github.com/Artoriuz/ArtCNN
- RIFE: https://github.com/hzwer/ECCV2022-RIFE
- RT64: https://github.com/rt64/rt64
- Anime4K: https://github.com/bloc97/Anime4K
- SRGD dataset: https://github.com/epishchik/SRGD

### Datasets
- GameIR: https://huggingface.co/datasets/LLLebin/GameIR
- SRGD: https://huggingface.co/datasets/epishchik/SRGD
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/

### Optimization Tools
- TensorRT: https://developer.nvidia.com/tensorrt
- ONNX Runtime + DirectML: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- Brevitas (INT8 QAT): https://github.com/Xilinx/brevitas
- TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer
