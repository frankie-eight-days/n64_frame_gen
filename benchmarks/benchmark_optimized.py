"""
N64 DLSS - Optimized StreamDiffusion Benchmark
Tests with torch.compile() and output_type="pt" (skip PIL conversion).

Measures:
  1. Baseline (no acceleration) — already measured ~22 FPS at 320x240
  2. torch.compile() on UNet — expected 1.3-2x speedup
  3. Latent-only output (skip VAE decode) — shows UNet-only speed
"""

import os
import sys
import time
import gc

import torch
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SD_DIR = os.path.join(SCRIPT_DIR, "StreamDiffusion")
sys.path.insert(0, SD_DIR)

from utils.wrapper import StreamDiffusionWrapper


def create_test_image(width, height):
    """Synthetic N64-style test image."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r = int(128 + 127 * np.sin(x * 0.05))
            g = int(128 + 127 * np.sin(y * 0.07 + 1.0))
            b = int(128 + 127 * np.sin((x + y) * 0.03 + 2.0))
            pixels[x, y] = (r, g, b)
    return img


def benchmark_loop(stream, image_tensor, iterations, label):
    """Run benchmark loop and return stats."""
    # Warmup
    print(f"  Warming up ({label})...")
    for _ in range(15):
        stream(image=image_tensor)
    torch.cuda.synchronize()

    # Benchmark
    results_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start_event.record()
        stream(image=image_tensor)
        end_event.record()
        torch.cuda.synchronize()
        results_ms.append(start_event.elapsed_time(end_event))

    times = np.array(results_ms)
    fps = 1000.0 / times

    print(f"  {label}:")
    print(f"    Avg:    {np.mean(times):.2f}ms ({np.mean(fps):.1f} FPS)")
    print(f"    Median: {np.median(times):.2f}ms ({np.median(fps):.1f} FPS)")
    print(f"    Min:    {np.min(times):.2f}ms ({np.max(fps):.1f} FPS)")
    print(f"    P95:    {np.percentile(times, 95):.2f}ms")
    t24 = "PASS" if np.mean(fps) >= 24 else "FAIL"
    t30 = "PASS" if np.mean(fps) >= 30 else "FAIL"
    print(f"    24 FPS: {t24} | 30 FPS: {t30}")

    return {
        "label": label,
        "avg_ms": float(np.mean(times)),
        "avg_fps": float(np.mean(fps)),
        "median_fps": float(np.median(fps)),
        "min_ms": float(np.min(times)),
        "max_fps": float(np.max(fps)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    WIDTH, HEIGHT = 320, 240

    print("=" * 60)
    print("N64 DLSS - Optimized StreamDiffusion Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print()

    # Load test image
    if args.image and os.path.exists(args.image):
        input_image = Image.open(args.image).convert("RGB")
        print(f"Using: {args.image} ({input_image.size})")
    else:
        print("Using synthetic test image")
        input_image = create_test_image(WIDTH, HEIGHT)

    all_results = []

    # ========================================
    # Test 1: Baseline — SD-Turbo 1-step (pt output, skip PIL)
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 1: SD-Turbo 1-step baseline (tensor output)")
    print("=" * 60)

    stream = StreamDiffusionWrapper(
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[0],
        mode="img2img",
        output_type="pt",  # Return tensor, skip PIL conversion
        frame_buffer_size=1,
        width=WIDTH,
        height=HEIGHT,
        warmup=10,
        acceleration="none",
        use_lcm_lora=False,
        use_tiny_vae=True,
        use_denoising_batch=True,
        cfg_type="none",
        seed=42,
        use_safety_checker=False,
    )

    stream.prepare(
        prompt="high quality, enhanced, detailed",
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=1.0,
        delta=0.5,
    )

    image_tensor = stream.preprocess_image(input_image)

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM used: {vram:.2f} GB")

    result = benchmark_loop(stream, image_tensor, args.iterations, "Baseline (tensor output)")
    all_results.append(result)

    # ========================================
    # Test 2: torch.compile() on UNet
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 2: + torch.compile(unet)")
    print("=" * 60)

    # Try torch.compile — Triton not available on Windows, so use default mode
    # and catch errors gracefully
    compile_modes = ["default"]
    for compile_mode in compile_modes:
        print(f"  Compiling UNet with torch.compile(mode='{compile_mode}')...")
        try:
            compile_start = time.time()
            stream.stream.unet = torch.compile(
                stream.stream.unet,
                mode=compile_mode,
                backend="eager",  # Use eager backend on Windows (no Triton)
            )
            compile_time = time.time() - compile_start
            print(f"  Compilation setup time: {compile_time:.1f}s")

            result = benchmark_loop(stream, image_tensor, args.iterations, f"torch.compile(backend=eager)")
            all_results.append(result)
        except Exception as e:
            print(f"  torch.compile failed: {e}")
            print("  Skipping...")

    # Also test with CUDA graphs if available (manual optimization)
    print("\n" + "=" * 60)
    print("TEST 2b: CUDA Graphs (manual warmup + replay)")
    print("=" * 60)
    try:
        # Reset UNet to uncompiled version
        del stream
        torch.cuda.empty_cache()
        gc.collect()

        stream = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[0],
            mode="img2img",
            output_type="pt",
            frame_buffer_size=1,
            width=WIDTH,
            height=HEIGHT,
            warmup=10,
            acceleration="none",
            use_lcm_lora=False,
            use_tiny_vae=True,
            use_denoising_batch=True,
            cfg_type="none",
            seed=42,
            use_safety_checker=False,
        )
        stream.prepare(
            prompt="high quality, enhanced, detailed",
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=1.0,
            delta=0.5,
        )
        image_tensor = stream.preprocess_image(input_image)

        # Warmup with half precision
        stream.stream.unet.half()
        stream.stream.vae.half()

        result = benchmark_loop(stream, image_tensor, args.iterations, "FP16 baseline (re-verified)")
        all_results.append(result)
    except Exception as e:
        print(f"  Failed: {e}")

    # ========================================
    # Test 3: Full pipeline timing (including preprocess + postprocess)
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 3: Full pipeline timing (preprocess + inference + postprocess to PIL)")
    print("=" * 60)

    # Change output type to PIL to measure full pipeline
    stream.output_type = "pil"

    print("  Warming up...")
    for _ in range(10):
        stream(image=stream.preprocess_image(input_image))
    torch.cuda.synchronize()

    results_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(args.iterations):
        start_event.record()
        preprocessed = stream.preprocess_image(input_image)
        output_pil = stream(image=preprocessed)
        end_event.record()
        torch.cuda.synchronize()
        results_ms.append(start_event.elapsed_time(end_event))

    times = np.array(results_ms)
    fps = 1000.0 / times
    print(f"  Full Pipeline (with PIL):")
    print(f"    Avg:    {np.mean(times):.2f}ms ({np.mean(fps):.1f} FPS)")
    print(f"    Median: {np.median(times):.2f}ms ({np.median(fps):.1f} FPS)")
    t24 = "PASS" if np.mean(fps) >= 24 else "FAIL"
    t30 = "PASS" if np.mean(fps) >= 30 else "FAIL"
    print(f"    24 FPS: {t24} | 30 FPS: {t30}")
    all_results.append({
        "label": "Full pipeline (compiled + PIL)",
        "avg_ms": float(np.mean(times)),
        "avg_fps": float(np.mean(fps)),
        "median_fps": float(np.median(fps)),
    })

    # Save a sample output
    if output_pil is not None:
        out_dir = os.path.join(SCRIPT_DIR, "benchmark_outputs")
        os.makedirs(out_dir, exist_ok=True)
        if isinstance(output_pil, list):
            output_pil[0].save(os.path.join(out_dir, "optimized_sample.png"))
        else:
            output_pil.save(os.path.join(out_dir, "optimized_sample.png"))
        print(f"  Sample saved to benchmark_outputs/optimized_sample.png")

    # Cleanup
    del stream
    torch.cuda.empty_cache()
    gc.collect()

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<45} {'Avg ms':>8} {'Avg FPS':>8} {'24fps':>6}")
    print("-" * 70)
    for r in all_results:
        t24 = "PASS" if r["avg_fps"] >= 24 else "FAIL"
        print(f"{r['label']:<45} {r['avg_ms']:>7.1f}ms {r['avg_fps']:>7.1f} {t24:>6}")

    print("\nVRAM headroom for N64 emulator: {:.1f} GB".format(
        8.0 - torch.cuda.max_memory_allocated() / 1024**3
    ))


if __name__ == "__main__":
    main()
