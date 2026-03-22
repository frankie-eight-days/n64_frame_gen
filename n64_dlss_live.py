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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "StreamDiffusion"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "n64emulator"))

import torch
from utils.wrapper import StreamDiffusionWrapper

# ============================================================
# Presets
# ============================================================

PRESETS = {
    "Default Enhance": {
        "prompt": "high quality, enhanced, sharp, detailed, remastered",
        "t_index": 35,
        "delta": 0.5,
    },
    "Studio Ghibli": {
        "prompt": "studio ghibli style, anime background, lush detailed",
        "t_index": 32,
        "delta": 0.5,
    },
    "Watercolor": {
        "prompt": "watercolor painting, soft edges, flowing colors, artistic",
        "t_index": 33,
        "delta": 0.5,
    },
    "Oil Painting": {
        "prompt": "oil painting, thick brushstrokes, impressionist, painterly",
        "t_index": 33,
        "delta": 0.5,
    },
    "Synthwave": {
        "prompt": "neon synthwave, glowing edges, cyberpunk, purple blue",
        "t_index": 30,
        "delta": 0.5,
    },
    "Modern Game": {
        "prompt": "modern AAA video game, ray traced, photorealistic, unreal engine 5",
        "t_index": 35,
        "delta": 0.5,
    },
}

# ============================================================
# Diffusion Processor (background thread)
# ============================================================

class DiffusionProcessor:
    def __init__(self):
        self.stream = None
        self.enabled = False
        self.running = True

        self.prompt = PRESETS["Default Enhance"]["prompt"]
        self.t_index = PRESETS["Default Enhance"]["t_index"]
        self.delta = PRESETS["Default Enhance"]["delta"]

        self._pending = None
        self._pending_lock = threading.Lock()

        # Frame exchange (GIL makes ref assignment atomic)
        self.latest_raw_frame = None
        self.latest_enhanced_frame = None

        self.diff_fps = 0.0
        self.status = "Initializing..."

    def init_model(self):
        self.status = "Loading StreamDiffusion + TensorRT..."
        print(self.status)
        self.stream = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[self.t_index],
            mode="img2img",
            output_type="pil",
            frame_buffer_size=1,
            width=320,
            height=240,
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
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=1.0,
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

    def request_settings(self, prompt, t_index, delta):
        with self._pending_lock:
            self._pending = {"prompt": prompt, "t_index": t_index, "delta": delta}

    def _apply_pending(self):
        with self._pending_lock:
            if self._pending is None:
                return
            s = self._pending
            self._pending = None

        self.status = "Applying settings..."
        self.prompt = s["prompt"]
        self.t_index = s["t_index"]
        self.delta = s["delta"]

        self.stream.stream.t_list = [self.t_index]
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=1.0,
            delta=self.delta,
        )
        self.status = "Running"

    def set_raw_frame(self, frame):
        self.latest_raw_frame = frame.copy()

    def process_loop(self):
        count = 0
        t0 = time.time()

        while self.running:
            self._apply_pending()

            if not self.enabled or self.latest_raw_frame is None:
                time.sleep(0.005)
                continue

            try:
                pil_img = Image.fromarray(self.latest_raw_frame)
                tensor = self.stream.preprocess_image(pil_img)
                result = self.stream(image=tensor)
                if isinstance(result, list):
                    result = result[0]
                self.latest_enhanced_frame = np.array(result)

                count += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.diff_fps = count / (now - t0)
                    count = 0
                    t0 = now
            except Exception as e:
                self.status = f"Error: {e}"
                time.sleep(0.5)


# ============================================================
# tkinter Control Panel
# ============================================================

def create_control_panel(processor):
    root = tk.Tk()
    root.title("N64 DLSS Control Panel")
    root.geometry("300x640")
    root.resizable(False, True)

    enabled_var = tk.BooleanVar(value=False)
    prompt_var = tk.StringVar(value=processor.prompt)
    t_index_var = tk.IntVar(value=processor.t_index)
    delta_var = tk.DoubleVar(value=processor.delta)
    status_var = tk.StringVar(value="Ready")
    diff_fps_var = tk.StringVar(value="Diffusion: -- FPS")
    emu_fps_var = tk.StringVar(value="Emulator: -- FPS")

    # -- Toggle --
    f_ctrl = ttk.LabelFrame(root, text="Control", padding=8)
    f_ctrl.pack(fill="x", padx=8, pady=4)

    def on_toggle():
        processor.enabled = enabled_var.get()
        if processor.enabled:
            processor.status = "Running"
        else:
            processor.status = "Ready (diffusion OFF)"

    ttk.Checkbutton(f_ctrl, text="Enable Diffusion  [F2]",
                    variable=enabled_var, command=on_toggle).pack(anchor="w")

    # -- Settings --
    f_set = ttk.LabelFrame(root, text="Settings", padding=8)
    f_set.pack(fill="x", padx=8, pady=4)

    ttk.Label(f_set, text="Prompt:").pack(anchor="w")
    prompt_entry = ttk.Entry(f_set, textvariable=prompt_var, width=38)
    prompt_entry.pack(fill="x", pady=(0, 6))

    t_label = ttk.Label(f_set, text=f"t_index: {t_index_var.get()}")
    t_label.pack(anchor="w")

    def on_t_slide(val):
        t_label.config(text=f"t_index: {int(float(val))}")

    ttk.Scale(f_set, from_=0, to=49, variable=t_index_var,
              orient="horizontal", command=on_t_slide).pack(fill="x")

    hint = tk.Frame(f_set)
    hint.pack(fill="x")
    ttk.Label(hint, text="<- More effect", font=("", 7)).pack(side="left")
    ttk.Label(hint, text="Faithful ->", font=("", 7)).pack(side="right")

    tk.Frame(f_set, height=6).pack()

    d_label = ttk.Label(f_set, text=f"Delta: {delta_var.get():.2f}")
    d_label.pack(anchor="w")

    def on_d_slide(val):
        d_label.config(text=f"Delta: {float(val):.2f}")

    ttk.Scale(f_set, from_=0.0, to=1.0, variable=delta_var,
              orient="horizontal", command=on_d_slide).pack(fill="x")

    def apply_settings():
        processor.request_settings(
            prompt=prompt_var.get(),
            t_index=int(t_index_var.get()),
            delta=float(delta_var.get()),
        )

    ttk.Button(f_set, text="Apply Settings",
               command=apply_settings).pack(fill="x", pady=(8, 0))
    prompt_entry.bind("<Return>", lambda e: apply_settings())

    # -- Presets --
    f_pre = ttk.LabelFrame(root, text="Presets", padding=8)
    f_pre.pack(fill="x", padx=8, pady=4)

    def apply_preset(name):
        p = PRESETS[name]
        prompt_var.set(p["prompt"])
        t_index_var.set(p["t_index"])
        delta_var.set(p["delta"])
        on_t_slide(p["t_index"])
        on_d_slide(p["delta"])
        apply_settings()

    for name in PRESETS:
        ttk.Button(f_pre, text=name,
                   command=lambda n=name: apply_preset(n)).pack(fill="x", pady=1)

    # -- Status --
    f_stat = ttk.LabelFrame(root, text="Status", padding=8)
    f_stat.pack(fill="x", padx=8, pady=4)

    ttk.Label(f_stat, textvariable=diff_fps_var).pack(anchor="w")
    ttk.Label(f_stat, textvariable=emu_fps_var).pack(anchor="w")
    ttk.Label(f_stat, textvariable=status_var).pack(anchor="w")

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

    # ---- Init tkinter control panel ----
    root = create_control_panel(processor)

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

        # -- Swap in enhanced frame if diffusion is on --
        if processor.enabled and processor.latest_enhanced_frame is not None:
            frontend.frame_data = processor.latest_enhanced_frame

        # -- Draw using frontend's GL renderer --
        if frontend.hw_render:
            pass  # core already rendered to GL
        else:
            frontend._draw_frame_gl(disp_w, disp_h)

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
