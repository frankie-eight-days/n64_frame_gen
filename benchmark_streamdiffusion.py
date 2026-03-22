"""
N64 DLSS - StreamDiffusion Benchmark
Measures real-time img2img performance on GTX 4060 at various resolutions.

Tests:
  1. 320x240 (native N64 resolution)
  2. 512x512 (standard SD resolution for comparison)
  3. 640x480 (2x N64)

Models:
  - SD-Turbo (1-step, optimized for speed)
  - SD 1.5 + LCM-LoRA (2-step and 4-step)

Usage:
  python benchmark_streamdiffusion.py [--image path/to/image.png]
"""

import os
import sys
import time
import json
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# Add StreamDiffusion to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SD_DIR = os.path.join(SCRIPT_DIR, "StreamDiffusion")
sys.path.insert(0, SD_DIR)

from utils.wrapper import StreamDiffusionWrapper


def create_test_image(width, height):
    """Create a synthetic N64-style test image if no real image is provided."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r = int(128 + 127 * np.sin(x * 0.05))
            g = int(128 + 127 * np.sin(y * 0.07 + 1.0))
            b = int(128 + 127 * np.sin((x + y) * 0.03 + 2.0))
            pixels[x, y] = (r, g, b)
    return img


def run_benchmark(config, input_image, iterations=50, warmup=10):
    """Run a single benchmark configuration and return results."""
    print(f"\n{'='*60}")
    print(f"Config: {config['name']}")
    print(f"  Model: {config['model']}")
    print(f"  Resolution: {config['width']}x{config['height']}")
    print(f"  Steps: {len(config['t_index_list'])}")
    print(f"  Acceleration: {config['acceleration']}")
    print(f"{'='*60}")

    # Resize input image to target resolution
    test_img = input_image.resize((config["width"], config["height"]), Image.LANCZOS)

    try:
        print("Loading model...")
        load_start = time.time()

        wrapper_kwargs = dict(
            model_id_or_path=config["model"],
            t_index_list=config["t_index_list"],
            mode="img2img",
            output_type="pil",
            frame_buffer_size=1,
            width=config["width"],
            height=config["height"],
            warmup=warmup,
            acceleration=config["acceleration"],
            use_lcm_lora=config.get("use_lcm_lora", True),
            use_tiny_vae=True,
            enable_similar_image_filter=False,
            use_denoising_batch=True,
            cfg_type=config.get("cfg_type", "none"),
            seed=42,
            use_safety_checker=False,
        )

        stream = StreamDiffusionWrapper(**wrapper_kwargs)

        stream.prepare(
            prompt=config.get("prompt", "high quality, detailed, sharp"),
            negative_prompt="blurry, low quality",
            num_inference_steps=50,
            guidance_scale=config.get("guidance_scale", 1.0),
            delta=config.get("delta", 0.5),
        )

        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s")

        # Get VRAM usage after loading
        vram_allocated = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved")

        # Warmup
        print(f"Warming up ({warmup} iterations)...")
        image_tensor = stream.preprocess_image(test_img)
        for _ in range(warmup):
            stream(image=image_tensor)
        torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking ({iterations} iterations)...")
        results_ms = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        output_image = None
        for i in range(iterations):
            start_event.record()
            image_tensor = stream.preprocess_image(test_img)
            output_image = stream(image=image_tensor)
            end_event.record()
            torch.cuda.synchronize()
            results_ms.append(start_event.elapsed_time(end_event))

        # Save one sample output
        if output_image is not None:
            out_dir = os.path.join(SCRIPT_DIR, "benchmark_outputs")
            os.makedirs(out_dir, exist_ok=True)
            safe_name = config["name"].replace(" ", "_").replace("/", "_")
            output_image.save(os.path.join(out_dir, f"{safe_name}.png"))

        # Calculate stats
        times = np.array(results_ms)
        fps_arr = 1000.0 / times

        stats = {
            "name": config["name"],
            "model": config["model"],
            "resolution": f"{config['width']}x{config['height']}",
            "steps": len(config["t_index_list"]),
            "acceleration": config["acceleration"],
            "avg_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "std_ms": float(np.std(times)),
            "avg_fps": float(np.mean(fps_arr)),
            "median_fps": float(np.median(fps_arr)),
            "min_fps": float(np.min(fps_arr)),
            "max_fps": float(np.max(fps_arr)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "vram_allocated_gb": round(vram_allocated, 2),
            "vram_reserved_gb": round(vram_reserved, 2),
            "load_time_s": round(load_time, 1),
        }

        print(f"\nResults:")
        print(f"  Avg:    {stats['avg_ms']:.2f}ms ({stats['avg_fps']:.1f} FPS)")
        print(f"  Median: {stats['median_ms']:.2f}ms ({stats['median_fps']:.1f} FPS)")
        print(f"  Min:    {stats['min_ms']:.2f}ms ({stats['max_fps']:.1f} FPS)")
        print(f"  Max:    {stats['max_ms']:.2f}ms ({stats['min_fps']:.1f} FPS)")
        print(f"  P95:    {stats['p95_ms']:.2f}ms")
        print(f"  P99:    {stats['p99_ms']:.2f}ms")
        target_24 = "PASS" if stats["avg_fps"] >= 24 else "FAIL"
        target_30 = "PASS" if stats["avg_fps"] >= 30 else "FAIL"
        target_60 = "PASS" if stats["avg_fps"] >= 60 else "FAIL"
        print(f"  24 FPS target: {target_24}")
        print(f"  30 FPS target: {target_30}")
        print(f"  60 FPS target: {target_60}")

        # Cleanup
        del stream
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        return stats

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return {"name": config["name"], "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="StreamDiffusion Benchmark for N64 DLSS")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (320x240 N64 screenshot)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--quick", action="store_true", help="Run only the most important configs")
    args = parser.parse_args()

    print("=" * 60)
    print("N64 DLSS - StreamDiffusion Benchmark")
    print("=" * 60)

    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Load or create test image
    if args.image and os.path.exists(args.image):
        print(f"Using input image: {args.image}")
        input_image = Image.open(args.image).convert("RGB")
        print(f"  Original size: {input_image.size}")
    else:
        if args.image:
            print(f"Image not found: {args.image}")
        print("Using synthetic test image (provide --image for real N64 screenshot)")
        input_image = create_test_image(320, 240)

    # Define benchmark configurations
    configs = []

    # === SD-Turbo at N64 resolution (320x240) — the primary test ===
    # SD-Turbo needs dimensions divisible by 8, 320x240 is fine
    configs.append({
        "name": "SD-Turbo 320x240 1-step",
        "model": "stabilityai/sd-turbo",
        "width": 320,
        "height": 240,
        "t_index_list": [0],
        "acceleration": "none",
        "use_lcm_lora": False,
        "cfg_type": "none",
        "guidance_scale": 1.0,
        "prompt": "high quality, detailed, sharp, enhanced",
    })

    # === SD-Turbo at 512x512 for comparison ===
    configs.append({
        "name": "SD-Turbo 512x512 1-step",
        "model": "stabilityai/sd-turbo",
        "width": 512,
        "height": 512,
        "t_index_list": [0],
        "acceleration": "none",
        "use_lcm_lora": False,
        "cfg_type": "none",
        "guidance_scale": 1.0,
        "prompt": "high quality, detailed, sharp, enhanced",
    })

    if not args.quick:
        # === SD-Turbo at 640x480 (2x N64) ===
        configs.append({
            "name": "SD-Turbo 640x480 1-step",
            "model": "stabilityai/sd-turbo",
            "width": 640,
            "height": 480,
            "t_index_list": [0],
            "acceleration": "none",
            "use_lcm_lora": False,
            "cfg_type": "none",
            "guidance_scale": 1.0,
            "prompt": "high quality, detailed, sharp, enhanced",
        })

        # === SD 1.5 + LCM-LoRA, 2-step at 320x240 ===
        configs.append({
            "name": "SD1.5+LCM 320x240 2-step",
            "model": "KBlueLeaf/kohaku-v2.1",
            "width": 320,
            "height": 240,
            "t_index_list": [0, 16],
            "acceleration": "none",
            "use_lcm_lora": True,
            "cfg_type": "none",
            "guidance_scale": 1.0,
            "prompt": "high quality, detailed, sharp, enhanced",
        })

        # === SD 1.5 + LCM-LoRA, 4-step at 320x240 ===
        configs.append({
            "name": "SD1.5+LCM 320x240 4-step",
            "model": "KBlueLeaf/kohaku-v2.1",
            "width": 320,
            "height": 240,
            "t_index_list": [0, 16, 32, 45],
            "acceleration": "none",
            "use_lcm_lora": True,
            "cfg_type": "none",
            "guidance_scale": 1.0,
            "prompt": "high quality, detailed, sharp, enhanced",
        })

    # Run benchmarks
    all_results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config['name']}")
        result = run_benchmark(config, input_image, args.iterations, args.warmup)
        all_results.append(result)

    # Print summary
    print("\n")
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Avg ms':>8} {'Avg FPS':>8} {'Med FPS':>8} {'VRAM':>6} {'24fps':>6}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<35} {'ERROR':>8}")
        else:
            target = "PASS" if r["avg_fps"] >= 24 else "FAIL"
            print(f"{r['name']:<35} {r['avg_ms']:>7.1f}ms {r['avg_fps']:>7.1f} {r['median_fps']:>7.1f} {r['vram_allocated_gb']:>5.1f}G {target:>6}")

    # Save results to JSON
    output_file = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "iterations": args.iterations,
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
