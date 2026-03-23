"""
N64 DLSS - StreamDiffusion + TensorRT Benchmark
First run builds TRT engines (takes a few minutes), then measures FPS.
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
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r = int(128 + 127 * np.sin(x * 0.05))
            g = int(128 + 127 * np.sin(y * 0.07 + 1.0))
            b = int(128 + 127 * np.sin((x + y) * 0.03 + 2.0))
            pixels[x, y] = (r, g, b)
    return img


def benchmark(stream, image_tensor, iterations, label):
    # Warmup
    print(f"  Warming up...")
    for _ in range(15):
        stream(image=image_tensor)
    torch.cuda.synchronize()

    results_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start.record()
        stream(image=image_tensor)
        end.record()
        torch.cuda.synchronize()
        results_ms.append(start.elapsed_time(end))

    times = np.array(results_ms)
    fps = 1000.0 / times

    print(f"  {label}:")
    print(f"    Avg:    {np.mean(times):.2f}ms ({np.mean(fps):.1f} FPS)")
    print(f"    Median: {np.median(times):.2f}ms ({np.median(fps):.1f} FPS)")
    print(f"    Min:    {np.min(times):.2f}ms ({np.max(fps):.1f} FPS)")
    print(f"    P95:    {np.percentile(times, 95):.2f}ms")
    t24 = "PASS" if np.mean(fps) >= 24 else "FAIL"
    t30 = "PASS" if np.mean(fps) >= 30 else "FAIL"
    t60 = "PASS" if np.mean(fps) >= 60 else "FAIL"
    print(f"    24 FPS: {t24} | 30 FPS: {t30} | 60 FPS: {t60}")
    return {"label": label, "avg_ms": float(np.mean(times)), "avg_fps": float(np.mean(fps)), "median_fps": float(np.median(fps))}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    WIDTH, HEIGHT = 320, 240

    print("=" * 60)
    print("N64 DLSS - TensorRT Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Resolution: {WIDTH}x{HEIGHT}")

    if args.image and os.path.exists(args.image):
        input_image = Image.open(args.image).convert("RGB")
        print(f"Using: {args.image}")
    else:
        print("Using synthetic test image")
        input_image = create_test_image(WIDTH, HEIGHT)

    all_results = []

    # ========================================
    # Test 1: TensorRT acceleration
    # ========================================
    print("\n" + "=" * 60)
    print("Building TensorRT engines (first run only, may take 3-5 min)...")
    print("=" * 60)

    build_start = time.time()

    stream = StreamDiffusionWrapper(
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[0],
        mode="img2img",
        output_type="pt",
        frame_buffer_size=1,
        width=WIDTH,
        height=HEIGHT,
        warmup=10,
        acceleration="tensorrt",
        use_lcm_lora=False,
        use_tiny_vae=True,
        use_denoising_batch=True,
        cfg_type="none",
        seed=42,
        use_safety_checker=False,
        engine_dir=os.path.join(SCRIPT_DIR, "engines"),
    )

    stream.prepare(
        prompt="high quality, enhanced, detailed",
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=1.0,
        delta=0.5,
    )

    build_time = time.time() - build_start
    print(f"Engine build/load time: {build_time:.1f}s")

    image_tensor = stream.preprocess_image(input_image)
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM used: {vram:.2f} GB")

    result = benchmark(stream, image_tensor, args.iterations, "SD-Turbo + TensorRT 320x240")
    all_results.append(result)

    # Save a sample output
    stream.output_type = "pil"
    sample = stream(image=stream.preprocess_image(input_image))
    out_dir = os.path.join(SCRIPT_DIR, "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(sample, list):
        sample[0].save(os.path.join(out_dir, "tensorrt_sample.png"))
    else:
        sample.save(os.path.join(out_dir, "tensorrt_sample.png"))
    print(f"  Sample saved to benchmark_outputs/tensorrt_sample.png")

    # Cleanup
    del stream
    torch.cuda.empty_cache()
    gc.collect()

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Config':<40} {'Avg ms':>8} {'Avg FPS':>8}")
    print("-" * 60)
    # Include baseline from previous runs for comparison
    print(f"{'Baseline (no accel, from prev run)':<40} {'~41.7ms':>8} {'~24.0':>8}")
    for r in all_results:
        print(f"{r['label']:<40} {r['avg_ms']:>7.1f}ms {r['avg_fps']:>7.1f}")

    speedup = 41.7 / all_results[0]["avg_ms"] if all_results else 0
    print(f"\nTensorRT speedup vs baseline: {speedup:.2f}x")
    print(f"VRAM headroom: {8.0 - torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
