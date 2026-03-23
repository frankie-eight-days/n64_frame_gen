"""
Test different artistic styles using the two-step approach (t_index_list=[35, 45])
which gave the best structure preservation with visible stylization.
"""
import os, sys, time
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SD_DIR = os.path.join(SCRIPT_DIR, "StreamDiffusion")
sys.path.insert(0, SD_DIR)

from utils.wrapper import StreamDiffusionWrapper

WIDTH, HEIGHT = 320, 240

img = Image.open(os.path.join(SCRIPT_DIR, "test.jpg")).convert("RGB")
img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)

out_dir = os.path.join(SCRIPT_DIR, "test_outputs")
os.makedirs(out_dir, exist_ok=True)

styles = [
    ("watercolor painting, soft edges, flowing colors, artistic", "watercolor"),
    ("oil painting, thick brushstrokes, impressionist, painterly", "oil_paint"),
    ("studio ghibli style, anime background, lush detailed", "ghibli"),
    ("modern AAA video game, ray traced, photorealistic, unreal engine 5", "modern_game"),
    ("pixel art, retro 16bit, snes style, crisp pixels", "pixel_art"),
    ("claymation, stop motion, plasticine, clay figures", "claymation"),
    ("neon synthwave, glowing edges, cyberpunk, purple blue", "synthwave"),
    ("pencil sketch, hand drawn, crosshatching, detailed linework", "sketch"),
]

# Use the two-step approach that showed best balance
for prompt, label in styles:
    print(f"\n=== {label} ===")
    stream = StreamDiffusionWrapper(
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[35, 45],
        mode="img2img",
        output_type="pil",
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
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=1.0,
        delta=0.5,
    )

    image_tensor = stream.preprocess_image(img)
    for _ in range(5):
        stream(image=image_tensor)
    result = stream(image=image_tensor)
    if isinstance(result, list):
        result = result[0]
    result.save(os.path.join(out_dir, f"style_{label}.png"))
    print(f"  Saved style_{label}.png")

    del stream
    torch.cuda.empty_cache()

print(f"\nAll outputs in {out_dir}/")
